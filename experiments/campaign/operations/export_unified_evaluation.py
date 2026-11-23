#!/usr/bin/env python3
"""Stage new-protocol tables and ID advice after all six campaign audits pass.

Never updates the paper or experiment artifacts. --check is read-only; --wait
may be supervised by an operational CPU job. No CUDA/model imports are needed.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = Path(__file__).resolve().parent / 'export_support'
PROTOCOL = 'unified_raw_ba_v1'
ID_GUIDANCE_PATH = 'data/current_id_space_guidance.jsonl'
METHODS = [('baselines', 'H1_rule'), ('baselines', 'H2_rule'),
           ('single_baselines', 'S_VINE'), ('single_baselines', 'S_TrustMark'),
           ('single_baselines', 'S_VideoSeal')]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def verify_hashes(mapping, checked):
    require(isinstance(mapping, dict) and mapping, 'missing evidence/source manifest')
    for name, expected in mapping.items():
        observed = checked.setdefault(name, digest(name)) if name not in checked else checked[name]
        require(observed == expected, 'frozen evidence changed: ' + name)


def completion_gate(root):
    """Inspect complete lane evidence before any report rows are consumed."""
    root = Path(root).resolve()
    locations = [root] + [root / folder / method for folder, method in METHODS]
    pending, audits, checked = [], {}, {}
    for location in locations:
        path = location / 'pipeline_audit.json'
        if not path.exists():
            pending.append(str(path))
            continue
        audit = read(path)
        require(audit.get('passed') is True, 'lane audit failed: ' + str(path))
        if location == root:
            require(audit.get('acceptance_protocol') == PROTOCOL,
                    'main audit uses another acceptance protocol')
            require(audit.get('original_requests') == 10000 and
                    audit.get('arm_request_evaluations') == 20000 and
                    audit.get('independent_final_test') is True and
                    audit.get('actual_latency_measurements_complete') is True and
                    audit.get('final_feedback_to_selection') is False,
                    'main audit lacks full frozen final completion')
            require(audit.get('heldout_reused_across_protocol_revisions') is True and
                    audit.get('newly_unseen_holdout_claim') is False,
                    'holdout-reuse scope not acknowledged in main audit')
            report = location / 'final_results.json'
        else:
            require(audit.get('complete') is True and audit.get('terminal_requests') == 20000
                    and audit.get('no_final_feedback') is True,
                    'baseline audit incomplete: ' + str(path))
            report = location / 'final/report.json'
        require(str(report) in audit.get('evidence', {}), 'report is not bound by lane audit')
        audits[str(path)] = dict(sha256=digest(path), audit=audit)
    # Missing audits are an ordinary pending state. Do not read partial reports.
    if pending:
        return dict(ready=False, pending=pending, audits=audits)
    for item in audits.values():
        verify_hashes(item['audit']['evidence'], checked)
    verify_hashes(read(root / 'source_hashes.json'), checked)
    checked[str(root / 'source_hashes.json')] = digest(root / 'source_hashes.json')
    for folder in ('baselines', 'single_baselines'):
        path = root / folder / 'protocol_sources.json'
        verify_hashes(read(path)['hashes'], checked)
        checked[str(path)] = digest(path)
        policy_path = root / folder / 'policy_freeze.json'
        policy = read(policy_path)
        verify_hashes(policy['source_hashes'], checked)
        checked[str(policy_path)] = digest(policy_path)
        for method, spec in policy['methods'].items():
            require(spec['rows'] == 20000, 'baseline proposal denominator changed')
            verify_hashes({spec['path']: spec['sha256']}, checked)
    return dict(ready=True, pending=[], audits=audits, evidence=checked)


def support_module():
    """Use a recorded local copy, independent of later paper-source edits."""
    for name, item in read(SUPPORT / 'source_copy.json').items():
        require(digest(SUPPORT / name) == item.get('adapted_sha256', item['sha256']), 'export support changed: ' + name)
    sys.path.insert(0, str(SUPPORT))
    spec = importlib.util.spec_from_file_location('_unified_export_support', SUPPORT / 'export_current_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lane(module, root, is_main, inputs):
    frozen, rows, splits, plan = module.load_frozen(root, inputs)
    require(frozen.get('acceptance_protocol') == PROTOCOL,
            'legacy frozen selection cannot enter unified tables: ' + str(root))
    ready, audit = module.audit_ready(root, is_main, inputs)
    require(ready, 'audit disappeared or became incomplete during export')
    report_path = root / ('final_results.json' if is_main else 'final/report.json')
    def validate_protocol_row(row):
        if not is_main and row['final_status'] == 'measured':
            require(row.get('acceptance_protocol') == PROTOCOL, 'legacy baseline final measurement row')
    final, metadata = module.load_report(report_path, frozen, rows, plan, inputs,
                                          row_validator=validate_protocol_row)
    if is_main:
        require(metadata.get('acceptance_protocol') == PROTOCOL, 'legacy main final report')
    else:
        # Older baseline report schema has no protocol field. Its new-protocol
        # frozen plan, audited source closure and exact result hashes bind it.
        require(metadata.get('two_arms_are_paired_duplicates_for_rule') is True,
                'baseline arms must explicitly share observations')
    return frozen, rows, splits, plan, final, audit


CAPTIONS = {
    'tab/main.tex': r'\caption{\textbf{Unified raw-bit acceptance: full-request results.} Each row uses all 10,000 requests, including requests without a deployed configuration. Selected counts terminal choices accepted on selection images; final success additionally requires the registered held-out detection, PSNR and actual latency gates. All six campaign audits passed before export; an audit certifies evidence completeness, not universal success. Baseline arms share observations and are counted once. Single-fragment rows use strength 1.0 and no frontend. The held-out benchmark is disjoint from selection and calibration but reused across protocol revisions; it is not newly unseen confirmation.}',
    'tab/classes.tex': r'\caption{\textbf{Unified raw-bit acceptance: held-out success by class.} Cells show count (percentage) with the full 2,000-request denominator, including non-deployments. All methods use identical frozen requests and image records. Baseline duplicate arms are counted once. These are completed results for the revised acceptance protocol on the previously used held-out benchmark, not a fresh confirmation set.}',
    'tab/main_quality.tex': r'\caption{\textbf{Unified raw-bit acceptance: observed quality and paired differences.} Own-pass means condition on each method\textquotesingle s successful unique requests; their conditioning sets differ. Paired differences use only the same requests passed by both methods. PSNR uses all 100 held-out cover/embedded pairs. Capacity is the worst-requested-attack BSC estimate from best-fragment BA, not demonstrated payload. Latency is the maximum requested-attack mean embedding plus actual single-FPR decoder time on an A100-SXM4-80GB; attacks, initialization and I/O are excluded. Where certified embedding measurements are reused, this is a cross-session component sum, not a newly measured monolithic call. It is not a tail-latency guarantee.}',
}


def revised_caption(content, caption):
    lines = content.splitlines()
    require(sum(line.startswith(r'\caption{') for line in lines) == 1, 'ambiguous table caption')
    return '\n'.join(caption if line.startswith(r'\caption{') else
                     '% Generated by operations/export_unified_evaluation.py; staged only.'
                     if line.startswith('% Generated') else line for line in lines) + '\n'


def assemble(root, gate):
    module = support_module()
    inputs = module.Inputs()
    frozen, rows, splits, plan, main_final, main_audit = load_lane(module, root, True, inputs)
    selection_audit = inputs.read(root / 'fallback/pipeline_audit.json')
    require(selection_audit.get('passed') is True and selection_audit.get('arm_request_evaluations') == 20000,
            'incomplete main selection audit')
    data = {f'main{arm}': module.method_summary(f'main{arm}', arm, rows, main_final,
            len(plan['tasks']), main_audit) for arm in module.ARMS}
    guidance_records = [record for arm in module.ARMS
        for record in module.id_guidance_rows(f'main{arm}', arm, rows, main_final)]
    baseline_results, equality = {}, {}
    singles_policy = inputs.read(root / 'single_baselines/policy_freeze.json')
    for folder, method in METHODS:
        location = root / folder / method
        bf, selected, bs, bp, final, audit = load_lane(module, location, False, inputs)
        require(bs == splits, 'baseline image records differ: ' + method)
        require({key: r['inputs'] for key, r in selected.items()} ==
                {key: r['inputs'] for key, r in rows.items()}, 'baseline requests differ: ' + method)
        if method in module.SINGLE_FRAGMENTS:
            module.pending_single(method, location, singles_policy, rows, inputs)
            fragment = module.SINGLE_FRAGMENTS[method]
            cfg = dict(S=[fragment], order=[fragment], s={fragment: 1.0}, fe_on=[])
            require(all(r['selected'] is None or r['selected']['cfg'] == cfg for r in selected.values()),
                    'single-fragment frozen configuration changed')
        for cls in module.CLASSES:
            for index in range(2000):
                a, b = (cls, '04', index), (cls, '06', index)
                require(selected[a]['selected'] == selected[b]['selected'] and final[a] == final[b],
                        'baseline duplicate arms differ: ' + method)
        data[method] = module.method_summary(method, '04', selected, final, len(bp['tasks']), audit)
        guidance_records.extend(module.id_guidance_rows(method, '04', selected, final))
        baseline_results[method] = final
        equality[method] = dict(requests_exactly_equal=True, image_records_exactly_equal=True,
                                duplicate_arms_exactly_equal=True)
    pairs = module.paired_comparisons(main_final, baseline_results)
    for item in data.values():
        require(0 <= item['final_pass'] <= item['selected'] <= 10000, 'invalid request counts')
        require(sum(x['final_pass'] for x in item['by_class'].values()) == item['final_pass'],
                'class counts disagree with overall result')
        require(item['own_pass_quality']['conditional_count'] == item['final_pass'],
                'quality conditioning denominator changed')
    files = {'tab/main.tex': module.table_main(data), 'tab/classes.tex': module.table_classes(data),
             'tab/main_quality.tex': module.table_quality(data, pairs)}
    files = {name: revised_caption(value, CAPTIONS[name]) for name, value in files.items()}
    require(len(guidance_records) == 70000,
            'ID advice must retain all 10,000 requests in each of seven method/arm rows')
    reuse = inputs.read(root / 'operations/holdout_reuse_decision.json')
    require(reuse.get('newly_unseen_holdout_claim') is False and
            reuse.get('holdout_reused_across_protocol_revisions') is True, 'holdout scope changed')
    provenance = dict(schema='unified_raw_ba_paper_staging_v2', acceptance_protocol=PROTOCOL,
        exported_at=utc(), root=str(root), all_six_audits_passed=True,
        script_sha256=digest(__file__), support=read(SUPPORT/'source_copy.json'),
        methods=data, paired_comparisons=pairs, dataset_equality=equality,
        completion_gate=gate, inputs=inputs.files, holdout_reuse= reuse,
        notes=['Staging output only; no paper source or selector changed.',
               'All 10,000 requests per method/arm; 2,000 per class. Baseline duplicate arms collapsed.',
               'Raw sign-match BA acceptance uses the registered fixed worst-case test-count correction.',
               'No BCH identity/recovery branch is an acceptance route under this protocol.',
               'Theoretical FPR control is conditional on the declared fresh-query ideal-PRF null; not a measured universal/adaptive-attack guarantee.',
               'Main proposal search remains the registered single-frontend search scope; H2 retains its historical-rule multi-frontend capability. No frontier was expanded by export.',
               'H2 candidate thresholds are recomputed by fragment subset and frontend under the unified protocol; historical oracle attack-column coverage remains limited.',
               'SMT selection latency is modeled; baseline selection latency is measured.',
               'Final latency may combine certified unchanged embedding timing with remeasured decoder timing across sessions.',
               '90% per-attack empirical image passing is not a simultaneous population lower-bound guarantee.',
               'Capacity is a BSC estimate from best-fragment BA, not tested variable-length payload.',
               'ID advice uses each audited final-passing request capacity and original FPR; no length is selected automatically.',
               'An empty advice range does not change SMT status, live acceptance, final success, configuration, or encoding.',
               'Original holdout is reused across protocol revisions; unused confirmation is outside this run.',
               'No historical final count or partial new-protocol final count is inserted.'])
    return files, provenance, guidance_records


def export_once(root=ROOT):
    root = Path(root).resolve()
    gate = completion_gate(root)
    if not gate['ready']:
        return dict(complete=False, phase='waiting_for_lane_audits', pending=gate['pending'])
    files, provenance, guidance_records = assemble(root, gate)
    # Detect changes while parsing the reports, before publishing any table.
    after = completion_gate(root)
    require(after == gate, 'completion evidence changed during export')
    for name, item in provenance['inputs'].items():
        require(digest(name) == item['sha256'], 'export input changed during read: ' + name)
    identity = hashlib.sha256(json.dumps(dict(audits=gate['audits'],
        script=provenance['script_sha256'], support=provenance['support']), sort_keys=True).encode()).hexdigest()
    staging = root / 'operations/paper_staging'
    staging.mkdir(exist_ok=True)
    target = staging / identity[:20]
    if not target.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=staging))
        try:
            for name, content in files.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            provenance['outputs'] = {name: digest(temporary/name) for name in files}
            guidance = support_module().write_id_guidance(temporary, guidance_records, provenance['methods'])
            require(guidance['path'] == ID_GUIDANCE_PATH, 'unexpected ID-advice output path')
            guidance.update(acceptance_protocol=PROTOCOL,
                            capacity_source='audited_final_request_bottleneck_capacity',
                            affects_solver_or_final_verdict=False)
            provenance['id_space_guidance'] = guidance
            provenance['outputs'][ID_GUIDANCE_PATH] = guidance['sha256']
            atomic(temporary/'data/unified_evaluation_provenance.json', provenance)
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    else:
        existing = read(target/'data/unified_evaluation_provenance.json')
        require(existing['script_sha256'] == provenance['script_sha256'] and
                existing['completion_gate'] == gate, 'existing staging identity mismatch')
        verify_hashes({str(target/name): sha for name, sha in existing['outputs'].items()}, {})
    published = read(target/'data/unified_evaluation_provenance.json')
    guidance = published['id_space_guidance']
    require(guidance['path'] == ID_GUIDANCE_PATH and
            guidance['sha256'] == published['outputs'][ID_GUIDANCE_PATH],
            'ID-advice manifest is not bound to the published output')
    result = dict(complete=True, phase='staged', acceptance_protocol=PROTOCOL,
        snapshot=str(target), provenance=str(target/'data/unified_evaluation_provenance.json'),
        provenance_sha256=digest(target/'data/unified_evaluation_provenance.json'),
        id_space_guidance=dict(guidance, path=str(target/ID_GUIDANCE_PATH)),
        paper_sources_modified=False, paper_update_pending=True)
    atomic(staging/'latest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true', help='read-only completion gate; no report aggregation')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--poll', type=float, default=30)
    args = parser.parse_args()
    require(args.poll >= 1, 'poll must be at least one second')
    root = args.root.resolve()
    if args.check:
        status = completion_gate(root)
        print(json.dumps(dict(ready=status['ready'], pending=status['pending'])))
        return
    with (root/'operations/unified_export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = export_once(root)
            except Exception as exc:
                atomic(root/'operations/unified_export_state.json', dict(complete=False, phase='failed',
                    updated_at=utc(), error=repr(exc), job=os.environ.get('SLURM_JOB_ID')))
                raise
            atomic(root/'operations/unified_export_state.json', dict(result, updated_at=utc(),
                job=os.environ.get('SLURM_JOB_ID')))
            print(json.dumps(result), flush=True)
            if result['complete'] or not args.wait:
                return
            time.sleep(args.poll)


if __name__ == '__main__':
    main()
