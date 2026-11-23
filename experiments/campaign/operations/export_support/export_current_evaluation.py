#!/usr/bin/env python3
"""Export audited four-input campaign results; never fill from partial final tasks.

Run from any directory with Python 3.8+. The experiment directories are read-only
inputs. Writes three result tables, per-request ID-space advice, and provenance.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from id_space_guidance import blank_guidance, recommend_id_space

DEFAULT_MAIN = Path(os.environ.get('TAILOR_WORKSPACE', 'workspace') + '/rigor_editing_only_20260911')
DEFAULT_BASELINES = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/baselines')
DEFAULT_SINGLES = DEFAULT_BASELINES.parent / 'single_baselines_20260914'
SINGLE_FRAGMENTS = {'S_VINE': 'VINE', 'S_TrustMark': 'TrustMark', 'S_VideoSeal': 'VideoSeal'}
CLASSES = ('C1', 'C2', 'C3', 'C4', 'C5')
ARMS = ('04', '06')
FOUR_INPUTS = {'attacks', 'fpr', 'min_psnr_db', 'max_ms'}
LABELS = {'main04': r'SMT ($m_0=0.04$)', 'main06': r'SMT ($m_0=0.06$)',
          'H1_rule': 'H1: expert rule', 'H2_rule': 'H2: greedy rule',
          **{method: fragment + ' only' for method, fragment in SINGLE_FRAGMENTS.items()}}
METHODS = tuple(LABELS)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Inputs:
    def __init__(self):
        self.files = {}

    def track(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            self.files[key] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
        if expected is not None:
            require(self.files[key]['sha256'] == expected, 'evidence hash changed: ' + key)
        return self.files[key]['sha256']

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text())

    def evidence(self, audit):
        require(audit.get('evidence'), 'completed audit has no evidence hashes')
        for path, sha in audit['evidence'].items():
            self.track(path, sha)


def check_signed(value, field):
    require(value.get(field) == canonical_hash({k: v for k, v in value.items() if k != field}),
            'invalid signed artifact: ' + field)


def row_key(row):
    return row['class'], row['arm'], int(row['request_id'])


def text_key(key):
    return '/'.join(map(str, key))


def load_frozen(root, inputs):
    path = root / 'plans/final_selection.json'
    value = inputs.read(path)
    check_signed(value, 'selection_sha256')
    require(value['mode'] == 'final' and value['n'] == 100 and value['no_final_feedback'] is True,
            'not a frozen independent-final choice')
    require(value['original_requests'] == 10000 and value['arm_request_evaluations'] == 20000,
            'wrong full-request denominator')
    rows = {}
    counts = collections.Counter()
    for row in value['requests']:
        key = row_key(row)
        require(key not in rows and key[0] in CLASSES and key[1] in ARMS and 0 <= key[2] < 2000,
                'invalid or duplicate frozen request')
        require(set(row['inputs']) == FOUR_INPUTS, 'request does not have exactly four inputs')
        require('nfpa_sd21_xy40_s10_v1' not in row['inputs']['attacks'], 'excluded NFPA request')
        selected = row.get('selected')
        require((selected is not None) == row['terminal_status'].endswith('live_pass'),
                'nonterminal or inconsistent frozen choice')
        for source in (row.get('source'), None if selected is None else selected.get('source')):
            if source is not None:
                inputs.track(source['path'], source['sha256'])
        if selected is not None:
            require(selected['cfg_sha256'] == canonical_hash(selected['cfg']), 'configuration changed')
        rows[key] = row
        counts[key[:2]] += 1
    require(len(rows) == 20000 and len(counts) == 10 and set(counts.values()) == {2000},
            'incomplete frozen request population')
    splits_path = Path(value['split_manifest'])
    inputs.track(splits_path, value['split_manifest_sha256'])
    splits = inputs.read(splits_path)
    check_signed(splits, 'manifest_sha256')
    for role in ('selection', 'final'):
        split = splits['splits'][role]
        require(split['n'] == len(split['images']) == 100, 'wrong image sample count')
        require(split['images_sha256'] == canonical_hash(split['images']), 'image manifest changed')
    require(value['images_sha256'] == splits['splits']['final']['images_sha256'], 'wrong final images')
    plan = inputs.read(root / 'plans/final_tasks.json')
    check_signed(plan, 'plan_sha256')
    inputs.track(plan['frozen_path'], plan['frozen_file_sha256'])
    require(plan['selection_sha256'] == value['selection_sha256'], 'plan/frozen mismatch')
    expected = {}
    for key, row in rows.items():
        if row['selected'] is None:
            continue
        cfg = row['selected']['cfg_sha256']
        for attack in row['inputs']['attacks']:
            item = expected.setdefault((cfg, attack), {'fprs': set(), 'requests': set()})
            item['fprs'].add(row['inputs']['fpr'])
            item['requests'].add(text_key(key))
    actual = set()
    for task in plan['tasks']:
        check_signed(task, 'task_id')
        key = (task['cfg_sha256'], task['attack'])
        require(key in expected and key not in actual, 'unexpected/duplicate final task')
        actual.add(key)
        require(set(task['requests']) == expected[key]['requests'] and
                task['fprs'] == sorted(expected[key]['fprs']), 'incomplete task request/FPR coverage')
        require(task['images_sha256'] == value['images_sha256'] and task['mode'] == 'final',
                'task uses wrong images')
        require(task['n'] == (30 if task['attack'] == 'unmarker' else 100), 'reduced task sample')
    require(actual == set(expected), 'missing frozen final task')
    return value, rows, splits, plan


class JsonStream:
    """Stream a large top-level final report, retaining one request at a time."""
    def __init__(self, handle):
        self.handle, self.buffer, self.pos, self.eof = handle, '', 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        chunk = self.handle.read(1024 * 1024)
        self.eof = not chunk
        self.buffer += chunk

    def ws(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or self.eof:
                return
            self.fill()

    def token(self, expected):
        self.ws()
        require(self.pos < len(self.buffer) and self.buffer[self.pos] == expected,
                'invalid report JSON, expected ' + expected)
        self.pos += 1

    def peek(self):
        self.ws()
        return self.buffer[self.pos] if self.pos < len(self.buffer) else ''

    def value(self):
        self.ws()
        if self.pos > 1024 * 1024:
            self.buffer, self.pos = self.buffer[self.pos:], 0
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()
            else:
                # A number could end at a chunk boundary; extend before taking it.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.pos = end
                return value


def load_report(path, frozen, rows, plan, inputs, row_validator=None):
    inputs.track(path)
    compact, metadata = {}, {}
    with Path(path).open() as handle:
        stream = JsonStream(handle)
        stream.token('{')
        while stream.peek() != '}':
            key = stream.value()
            stream.token(':')
            if key == 'rows':
                stream.token('[')
                while stream.peek() != ']':
                    row = stream.value()
                    if row_validator is not None:
                        row_validator(row)
                    parts = row['key'].split('/')
                    rkey = (parts[0], parts[1], int(parts[2]))
                    require(rkey in rows and rkey not in compact, 'unexpected or duplicate final row')
                    require(row['inputs'] == rows[rkey]['inputs'], 'final report changed request inputs')
                    require(row['final_feedback_used'] is False, 'final feedback used')
                    deployed = rows[rkey]['selected'] is not None
                    require((row['final_status'] == 'measured') == deployed, 'report selection mismatch')
                    passed = row['final_pass']
                    require((isinstance(passed, bool) if deployed else passed is None), 'invalid final verdict')
                    result = {'final_pass': passed, 'final_status': row['final_status']}
                    if deployed:
                        require(row['cfg_sha256'] == rows[rkey]['selected']['cfg_sha256'], 'changed final configuration')
                        result.update(psnr_db=row['embedding_psnr_db']['mean'],
                            capacity_bits_estimate=row['capacity_bits_estimate'],
                            latency_ms=row['latency_max_attack_mean_ms'])
                        require(all(math.isfinite(float(result[k])) for k in
                                    ('psnr_db', 'capacity_bits_estimate', 'latency_ms')), 'invalid observed metric')
                        require(passed == bool(row['psnr_pass'] and row['latency_pass'] and
                                              row['empirical_detection_pass']), 'final gate summary mismatch')
                    compact[rkey] = result
                    if stream.peek() != ']':
                        stream.token(',')
                stream.token(']')
            else:
                value = stream.value()
                if key not in {'shared_attack_metrics', 'task_evidence'}:
                    metadata[key] = value
            if stream.peek() != '}':
                stream.token(',')
        stream.token('}')
    require(metadata['passed'] is True and metadata['complete'] is True and metadata['no_reselection'] is True,
            'final report is not audited and complete')
    require(metadata['selection_sha256'] == frozen['selection_sha256'] and
            metadata['plan_sha256'] == plan['plan_sha256'], 'final report provenance mismatch')
    require(metadata['original_requests'] == 10000 and metadata['arm_request_evaluations'] == 20000,
            'final report denominator changed')
    require(set(compact) == set(rows), 'incomplete final report population')
    require(sum(x['final_pass'] is True for x in compact.values()) == metadata['final_pass_requests'],
            'final count does not match rows')
    require(sum(x['final_status'] == 'measured' for x in compact.values()) == metadata['deployed_requests'],
            'deployed count does not match rows')
    return compact, metadata


def audit_ready(root, is_main, inputs):
    path = root / 'pipeline_audit.json'
    if not path.exists():
        return False, {'status': 'pending', 'reason': 'final pipeline audit absent', 'expected_path': str(path)}
    audit = inputs.read(path)
    ready = audit.get('passed') is True
    if is_main:
        ready = ready and audit.get('actual_latency_measurements_complete') is True and audit.get('independent_final_test') is True
    else:
        ready = ready and audit.get('complete') is True and audit.get('terminal_requests') == 20000
    if ready:
        inputs.evidence(audit)
    return ready, {'status': 'complete' if ready else 'pending', 'path': str(path),
                   'sha256': inputs.track(path), 'audit': audit}


def method_summary(name, arm, frozen_rows, final_rows, task_count, audit):
    keys = [key for key in frozen_rows if key[1] == arm]
    require(len(keys) == 10000, 'method has wrong unique-request denominator')
    selected = sum(frozen_rows[key]['selected'] is not None for key in keys)
    final_count = None if final_rows is None else sum(final_rows[key]['final_pass'] is True for key in keys)
    by_class = {}
    for cls in CLASSES:
        group = [key for key in keys if key[0] == cls]
        require(len(group) == 2000, 'wrong class denominator')
        count = None if final_rows is None else sum(final_rows[key]['final_pass'] is True for key in group)
        by_class[cls] = {'denominator': 2000, 'final_pass': count,
                         'final_rate': None if count is None else count / 2000}
    quality = None
    if final_rows is not None:
        passed = [final_rows[key] for key in keys if final_rows[key]['final_pass'] is True]
        quality = {'conditional_count': len(passed), 'aggregation': 'unweighted mean over own final-passing unique requests',
                   **{field: None if not passed else statistics.fmean(row[field] for row in passed)
                      for field in ('psnr_db', 'capacity_bits_estimate', 'latency_ms')}}
    return {'name': name, 'arm': arm, 'denominator': 10000, 'selected': selected,
            'selected_by_terminal_status': dict(collections.Counter(
                frozen_rows[key]['terminal_status'] for key in keys if frozen_rows[key]['selected'] is not None)),
            'not_deployed_by_terminal_status': dict(collections.Counter(
                frozen_rows[key]['terminal_status'] for key in keys if frozen_rows[key]['selected'] is None)),
            'selection_rate': selected / 10000, 'final_pass': final_count,
            'final_rate': None if final_count is None else final_count / 10000,
            'final_audit': audit['status'], 'by_class': by_class, 'own_pass_quality': quality,
            'unique_final_tasks_campaign': task_count}


def paired_comparisons(main_rows, baseline_rows):
    pairs = []
    if main_rows is None:
        return pairs
    for arm in ARMS:
        for baseline, results in baseline_rows.items():
            if results is None:
                continue
            keys = [key for key in main_rows if key[1] == arm and
                    main_rows[key]['final_pass'] is True and results[key]['final_pass'] is True]
            pairs.append({'main_arm': arm, 'baseline': baseline, 'common_final_pass_requests': len(keys),
                'difference_direction': 'SMT minus baseline, paired by the same class/request/FPR/budgets',
                **{field: None if not keys else statistics.fmean(main_rows[key][field] - results[key][field] for key in keys)
                   for field in ('psnr_db', 'capacity_bits_estimate', 'latency_ms')}})
    return pairs


def id_guidance_rows(method, arm, frozen_rows, final_rows, pending_selection=False):
    """Advise from each audited passing request; never from aggregate capacity."""
    for key in sorted(frozen_rows):
        if key[1] != arm:
            continue
        request = frozen_rows[key]
        selected = None if pending_selection else request['selected']
        final_pass = None if final_rows is None else final_rows[key]['final_pass']
        if pending_selection:
            advice = blank_guidance('pending_selection')
        elif selected is None:
            advice = blank_guidance('not_deployed')
        elif final_rows is None:
            advice = blank_guidance('pending_final_audit')
        elif final_pass is not True:
            advice = blank_guidance('final_failed')
        else:
            advice = recommend_id_space(final_rows[key]['capacity_bits_estimate'],
                                        request['inputs']['fpr'])
        yield {'method': method, 'class': key[0], 'arm': arm, 'request_id': key[2],
               'request_fpr': request['inputs']['fpr'], 'final_pass': final_pass,
               'cfg_sha256': None if selected is None else selected['cfg_sha256'],
               'guidance': advice}


def write_id_guidance(paper_root, records, methods):
    """Publish a complete report atomically and return its hash-bound manifest."""
    counts = {method: collections.Counter() for method in methods}
    seen = set()
    for row in records:
        key = (row['method'], row['class'], row['request_id'])
        require(key not in seen, 'duplicate ID-guidance request')
        seen.add(key)
        counts[row['method']][row['guidance']['status']] += 1
    for method, statuses in counts.items():
        require(sum(statuses.values()) == methods[method]['denominator'],
                'ID-guidance denominator mismatch: ' + method)
        passed = methods[method]['final_pass']
        if passed is not None:
            require(statuses['available'] + statuses['empty'] == passed,
                    'ID guidance was not restricted to audited final passes: ' + method)
    relative = 'data/current_id_space_guidance.jsonl'
    path = paper_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'schema': 'id_space_guidance_report_v1', 'path': relative,
            'sha256': file_hash(path), 'records': len(records),
            'helper_sha256': file_hash(Path(__file__).with_name('id_space_guidance.py')),
            'formula': 'id_bits_max = floor(capacity_bits_estimate + log2(fpr))',
            'model_guidance_only': True, 'selection_by': 'user',
            'range_deployment_validated': False,
            'evaluated_codeword_bits': 100, 'evaluated_payload_bits': 37,
            'by_method': {method: {'records': sum(statuses.values()), 'by_status': dict(statuses)}
                          for method, statuses in counts.items()}}


def pending_single(method, root, policy, main_selection, inputs):
    """A frozen proposal is not a live-accepted selection or a final result."""
    spec = policy['methods'][method]
    path = root / 'plans/proposals.json'
    require(str(path.resolve()) == spec['path'] and spec['rows'] == 20000,
            'single-fragment proposal scope changed')
    inputs.track(path, spec['sha256'])
    proposal = inputs.read(path)
    require(proposal['method'] == method, 'wrong single-fragment policy')
    expected_fragment = SINGLE_FRAGMENTS[method]
    expected_config = {'S': [expected_fragment], 'order': [expected_fragment],
                       's': {expected_fragment: 1.0}, 'fe_on': []}
    seen = set()
    for row in proposal['rows']:
        key = (row['group'], str(row['arm']), int(row['request_index']))
        require(key in main_selection and key not in seen, 'wrong single-fragment request population')
        seen.add(key)
        request = dict(row['request'])
        request['min_psnr_db'] = request.pop('min_psnr')
        require(request == main_selection[key]['inputs'], 'single-fragment request changed')
        require(row['config'] == expected_config, 'single-fragment setting is not the registered fixed rule')
    require(seen == set(main_selection), 'incomplete single-fragment proposals')
    return {'name': method, 'arm': '04', 'denominator': 10000, 'selected': None,
            'selection_rate': None, 'final_pass': None, 'final_rate': None,
            'final_audit': 'pending', 'own_pass_quality': None,
            'selected_by_terminal_status': {}, 'not_deployed_by_terminal_status': {},
            'unique_final_tasks_campaign': None,
            'status_reason': 'fixed proposals verified; complete live selection not yet frozen',
            'by_class': {cls: {'denominator': 2000, 'final_pass': None, 'final_rate': None}
                         for cls in CLASSES}}


def number(n):
    return f'{n:,}'


def pct(rate):
    return r'\textit{Pending}' if rate is None else f'{100 * rate:.2f}\\%'


def count(n):
    return r'\textit{Pending}' if n is None else number(n)


def format_metric(value):
    return '--' if value is None else f'{value:.2f}'


def table_main(data):
    lines = [r'% Generated by scripts/export_current_evaluation.py; do not hand-edit numbers.',
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{\textbf{Current four-input request results.} Each row uses all 10,000 requests (2,000 per class). Selected denotes a terminal configuration accepted on selection images: SMT includes its registered fallback, whereas each baseline has one rule proposal. These are not final successes. Final success requires detection, PSNR, and actual latency gates. Requests without a deployed configuration remain in the denominator. Final audit denotes evidence and process completeness, not universal success. Each baseline appears once because its two stored arms share the same policy and measurements; they are not independent repeats. Single-fragment rows fix strength at 1.0 with no frontend. Pending selection cells denote proposals that have not completed live selection. Pending final values are withheld until that method completes its final audit; both SMT arms await the joint main audit.}',
        r'\label{tab:main}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrl}', r'\toprule',
        r'Method & Selected & Selection rate & Final success & Final rate & Final audit \\', r'\midrule']
    for method in (method for method in METHODS if method in data):
        row = data[method]
        lines.append(f"{LABELS[method]} & {count(row['selected'])} & {pct(row['selection_rate'])} & {count(row['final_pass'])} & {pct(row['final_rate'])} & {row['final_audit'].capitalize()} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    return '\n'.join(lines)


def table_classes(data):
    lines = [r'% Generated by scripts/export_current_evaluation.py; full 2,000-request denominators.',
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{\textbf{Independent-final success by request class.} Each cell reports the successful request count (percentage). Every class has a denominator of 2,000, including requests for which a method did not deploy a configuration. All methods use the same frozen requests and the same image files. Baseline arms share measurements and are counted once. A method remains pending until its complete final audit; partial task completion is not reported as a class result.}',
        r'\label{tab:classes}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}', r'\toprule',
        r'Method & C1 & C2 & C3 & C4 & C5 \\', r'\midrule']
    for method in (method for method in METHODS if method in data):
        values = []
        for cls in CLASSES:
            row = data[method]['by_class'][cls]
            values.append(r'\textit{Pending}' if row['final_pass'] is None else
                          f"{number(row['final_pass'])} ({pct(row['final_rate'])})")
        lines.append(LABELS[method] + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    return '\n'.join(lines)


def table_quality(data, pairs):
    lines = [r'% Generated from complete independent-final reports only.',
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{\textbf{Observed quality on the requests each method passes at final.} Entries are request-weighted means; $n$ is the number of unique requests satisfying all final gates for that method. These conditioning sets differ, so their means are not a paired method comparison. PSNR uses the 100 final cover/embedded pairs. Capacity is the worst-requested-attack BSC estimate, not demonstrated coded payload. Latency is the maximum, over requested attacks, of measured mean embedding plus full request-FPR decoder time on an A100-SXM4-80GB, excluding attacks, initialization, and I/O. Each baseline is counted once. Quality for a method is shown only after its complete final audit; paired differences additionally require the corresponding SMT final audit.}',
        r'\label{tab:main-quality}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}', r'\toprule',
        r'Method & Own-pass $n$ & PSNR (dB) & Capacity (bits) & Latency (ms) \\', r'\midrule']
    for method in (method for method in METHODS if method in data):
        row = data[method]['own_pass_quality']
        if row is not None:
            lines.append(f"{LABELS[method]} & {number(row['conditional_count'])} & {format_metric(row['psnr_db'])} & {format_metric(row['capacity_bits_estimate'])} & {format_metric(row['latency_ms'])} " + r'\\')
    if pairs:
        lines += [r'\midrule', r'\multicolumn{5}{l}{\textit{Paired SMT minus baseline on common final-passing requests}} \\',
                  r'Comparison & Common $n$ & $\Delta$PSNR (dB) & $\Delta$capacity (bits) & $\Delta$latency (ms) \\']
        for pair in pairs:
            label = r'$m_0=0.' + pair['main_arm'] + '$ -- ' + LABELS[pair['baseline']]
            lines.append(f"{label} & {number(pair['common_final_pass_requests'])} & {format_metric(pair['psnr_db'])} & {format_metric(pair['capacity_bits_estimate'])} & {format_metric(pair['latency_ms'])} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    return '\n'.join(lines)


def export(main_root, baselines_root, paper_root, singles_root=DEFAULT_SINGLES):
    inputs = Inputs()
    frozen, main_selection, splits, plan = load_frozen(main_root, inputs)
    selection_audit = inputs.read(main_root / 'fallback/pipeline_audit.json')
    require(selection_audit['passed'] is True and selection_audit['arm_request_evaluations'] == 20000,
            'main selection audit incomplete')
    main_ready, main_audit = audit_ready(main_root, True, inputs)
    main_final, main_metadata = (load_report(main_root / 'final_results.json', frozen, main_selection, plan, inputs)
                                 if main_ready else (None, None))
    data = {f'main{arm}': method_summary(f'main{arm}', arm, main_selection, main_final, len(plan['tasks']), main_audit)
            for arm in ARMS}
    guidance = [row for arm in ARMS for row in
                id_guidance_rows(f'main{arm}', arm, main_selection, main_final)]
    baselines, audits, dataset = {}, {}, {}
    for method in ('H1_rule', 'H2_rule'):
        root = baselines_root / method
        bf, selection, bs, bp = load_frozen(root, inputs)
        require(bs == splits, 'baseline uses different image records')
        require({key: row['inputs'] for key, row in selection.items()} ==
                {key: row['inputs'] for key, row in main_selection.items()}, 'baseline uses different requests')
        ready, audit = audit_ready(root, False, inputs)
        final, metadata = (load_report(root / 'final/report.json', bf, selection, bp, inputs)
                           if ready else (None, None))
        for cls in CLASSES:
            for index in range(2000):
                a, c = (cls, '04', index), (cls, '06', index)
                require(selection[a]['selected'] == selection[c]['selected'], 'baseline policy arms differ')
                if final is not None:
                    require(final[a] == final[c], 'baseline measured arms differ')
        data[method] = method_summary(method, '04', selection, final, len(bp['tasks']), audit)
        guidance.extend(id_guidance_rows(method, '04', selection, final))
        baselines[method], audits[method] = final, audit
        dataset[method] = {'requests_exactly_equal': True, 'image_manifests_semantically_equal': True,
                           'arms_share_observations': True}
    single_policy_path = singles_root / 'policy_freeze.json'
    if single_policy_path.exists():
        policy = inputs.read(single_policy_path)
        for path, sha in policy['source_hashes'].items():
            inputs.track(path, sha)
        for method in SINGLE_FRAGMENTS:
            root = singles_root / method
            # Always verify the fixed single-fragment configuration and the full
            # request population, even after final selection becomes available.
            pending = pending_single(method, root, policy, main_selection, inputs)
            if not all((root / 'plans' / name).exists()
                       for name in ('final_selection.json', 'final_tasks.json')):
                data[method], baselines[method] = pending, None
                guidance.extend(id_guidance_rows(method, '04', main_selection, None,
                                                 pending_selection=True))
                audits[method] = {'status': 'pending', 'reason': pending['status_reason']}
                dataset[method] = {'requests_exactly_equal': True, 'arms_share_observations': True,
                                   'image_manifests_semantically_equal': None}
                split_path = root / 'plans/image_splits.json'
                if split_path.exists():
                    require(inputs.read(split_path) == splits, 'single-fragment image roles changed')
                    dataset[method]['image_manifests_semantically_equal'] = True
                continue
            sf, selection, ss, sp = load_frozen(root, inputs)
            require(ss == splits, 'single-fragment image roles changed')
            fragment = SINGLE_FRAGMENTS[method]
            expected_config = {'S': [fragment], 'order': [fragment],
                               's': {fragment: 1.0}, 'fe_on': []}
            require(all(row['selected'] is None or row['selected']['cfg'] == expected_config
                        for row in selection.values()), 'final single-fragment choice changed its fixed policy')
            require({key: row['inputs'] for key, row in selection.items()} ==
                    {key: row['inputs'] for key, row in main_selection.items()},
                    'single-fragment final requests changed')
            ready, audit = audit_ready(root, False, inputs)
            final, metadata = (load_report(root / 'final/report.json', sf, selection, sp, inputs)
                               if ready else (None, None))
            for cls in CLASSES:
                for index in range(2000):
                    a, b = (cls, '04', index), (cls, '06', index)
                    require(selection[a]['selected'] == selection[b]['selected'],
                            'single-fragment comparison arms differ')
                    if final is not None:
                        require(final[a] == final[b], 'single-fragment observations differ by arm')
            data[method] = method_summary(method, '04', selection, final, len(sp['tasks']), audit)
            guidance.extend(id_guidance_rows(method, '04', selection, final))
            baselines[method], audits[method] = final, audit
            dataset[method] = {'requests_exactly_equal': True, 'image_manifests_semantically_equal': True,
                               'arms_share_observations': True}
    pairs = paired_comparisons(main_final, baselines)
    # Arithmetic self-checks make denominator and own-pass conditioning explicit.
    for method, row in data.items():
        require(row['selected'] is None or 0 <= row['selected'] <= 10000, 'invalid selection count')
        if row['final_pass'] is not None:
            require(0 <= row['final_pass'] <= row['selected'], 'more final successes than deployments')
            require(sum(x['final_pass'] for x in row['by_class'].values()) == row['final_pass'], 'class sum mismatch')
            require(row['own_pass_quality']['conditional_count'] == row['final_pass'], 'quality denominator mismatch')
    files = {'tab/main.tex': table_main(data), 'tab/classes.tex': table_classes(data),
             'tab/main_quality.tex': table_quality(data, pairs)}
    guidance_manifest = write_id_guidance(paper_root, guidance, data)
    for name, content in files.items():
        path = paper_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    provenance = {'schema': 'paper_current_evaluation_export_v1',
        'exported_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_sha256': file_hash(__file__), 'main_root': str(main_root),
        'baselines_root': str(baselines_root), 'methods': data, 'paired_comparisons': pairs,
        'single_baselines_root': str(singles_root),
        'main_final_values_withheld': not main_ready, 'main_audit': main_audit,
        'baseline_audits': audits, 'dataset_equality': dataset,
        'id_space_guidance': guidance_manifest,
        'inputs': inputs.files, 'outputs': {name: file_hash(paper_root / name) for name in files},
        'notes': ['No partial main-final task data are read or reported.',
                  '10,000 unique requests per method/arm; 2,000 per class.',
                  'Baseline duplicate arms are collapsed after exact equality checks.',
                  'Own-pass quality means use different conditioning sets and are not paired comparisons.',
                  'Process-audit passed does not mean every request passed.',
                  'Selection latency differs: SMT model prediction, baseline measured latency.',
                  'Final latency is a maximum requested-attack mean, not a p95 guarantee.',
                  'Capacity is an observed BSC estimate, not demonstrated coded payload.',
                  'ID-length advice uses each audited final-passing request capacity and original FPR.',
                  'ID advice is a model budget, not measured variable-length recovery or false acceptance.',
                  'The user selects an ID length; export does not change the evaluated 100/37-bit codec.']}
    target = paper_root / 'data/current_evaluation_provenance.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'main_final_audited': main_ready,
        'counts': {k: {'selected': v['selected'], 'final_pass': v['final_pass']} for k, v in data.items()},
        'quality': {k: v['own_pass_quality'] for k, v in data.items()},
        'id_space_guidance': guidance_manifest,
        'provenance': str(target)}, indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-root', type=Path, default=DEFAULT_MAIN)
    parser.add_argument('--baselines-root', type=Path, default=DEFAULT_BASELINES)
    parser.add_argument('--single-baselines-root', type=Path, default=DEFAULT_SINGLES)
    parser.add_argument('--paper-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.main_root.resolve(), args.baselines_root.resolve(), args.paper_root.resolve(),
           args.single_baselines_root.resolve())


if __name__ == '__main__':
    main()
