"""Preregistered bridging check: may frozen cells stand in for measurements in the relocated root?

Fixed before any re-measurement (policy bridge_v1):

Sample. Population: the frozen cell cache (FROZEN/live_topk/cells), grouped by configuration. For each
fragment count k in {1, 2, 3}, configurations are ordered by sha256("bridge_v1|" + cfg_id). The first
configuration that has frozen cells for all 20 main attacks is re-measured on all 20; the next two that
have frozen cells for all 16 in-process main attacks are re-measured on those 16. Up to 9 configurations;
cross-environment attacks get up to 3. The images, decoder, key and thresholds are those of the frozen
cells (pool offset 300, n = 100, UnMarker 30).

Comparison. For every re-measured (configuration, attack) and every threshold bucket of the request alphas
present in both cells: rate = share of images accepted, mean = mean accepted score. Per attack, over all
its pairs and buckets: max |d rate|, max |d mean|, and the mean signed shifts.

Decision. Deterministic attacks (13: jpeg25, blur, bright, contrast, crop75, crop50, rot9, rs256, hflip,
crop_jpeg, border20, vaeB, vaeC) are approved when max |d rate| <= 0.03 and max |d mean| <= 0.01. Attacks
with unseeded randomness (7: noise, regen, rinse2x, ctrlregen_s03, ctrlregen_s05, unmarker,
editing_ip2p_s20_v1) are approved when |mean d rate| <= 0.03, |mean d mean| <= 0.015, max |d rate| <= 0.10
and max |d mean| <= 0.04. An attack without a complete comparison is not approved. Approved attacks may
reuse frozen cells; every other attack is measured in the run root. Rows measured under different
parameterizations of an attack are not combined in one table without re-measurement.

Usage (RUN from TAILOR_MAIN_CONTROLS_RUN): bridge.py select | submit [workers] | decide
"""
import collections
import hashlib
import json
import os
import subprocess
import sys
from common import (CP, FROZEN_CELLS, POLICY, RUN, APPROVAL, atomic, cell_name, load_cell, read, task_id)

BRIDGE_POLICY = 'bridge_v1'
IN_PROCESS = ('jpeg25', 'blur', 'noise', 'bright', 'contrast', 'crop75', 'crop50', 'rot9', 'rs256', 'hflip',
              'crop_jpeg', 'border20', 'vaeB', 'vaeC', 'regen', 'rinse2x')
CROSS_ENV = ('ctrlregen_s03', 'ctrlregen_s05', 'unmarker', 'editing_ip2p_s20_v1')
MAIN_ATTACKS = IN_PROCESS + CROSS_ENV
STOCHASTIC = frozenset({'noise', 'regen', 'rinse2x', 'ctrlregen_s03', 'ctrlregen_s05', 'unmarker',
                        'editing_ip2p_s20_v1'})
DETERMINISTIC = frozenset(MAIN_ATTACKS) - STOCHASTIC
TOL_DETERMINISTIC = dict(max_rate=0.03, max_mean=0.01)
TOL_STOCHASTIC = dict(shift_rate=0.03, shift_mean=0.015, max_rate=0.10, max_mean=0.04)
BRIDGE_CLS = 'C4'
DIR = RUN / 'bridging'
SUFFIX = '_pool300_n100.json'


def order_key(cfg_id):
    return hashlib.sha256(f'{BRIDGE_POLICY}|{cfg_id}'.encode()).hexdigest()


def frozen_population(cells_dir=FROZEN_CELLS):
    """cfg_id -> set of attacks with a frozen cell file."""
    by_cfg = collections.defaultdict(set)
    for name in os.listdir(cells_dir):
        if not name.endswith(SUFFIX):
            continue
        stem = name[:-len(SUFFIX)]
        by_cfg[stem[:64]].add(stem[65:])
    return by_cfg


def select_sample(by_cfg, fragment_count):
    """The preregistered sample from a population map and a cfg_id -> fragment count function."""
    chosen = []
    for k in (1, 2, 3):
        ids = sorted((c for c in by_cfg if fragment_count(c) == k), key=order_key)
        full = [c for c in ids if set(MAIN_ATTACKS) <= by_cfg[c]][:1]
        rest = [c for c in ids if set(IN_PROCESS) <= by_cfg[c] and c not in full][:2]
        chosen += [(c, k, list(MAIN_ATTACKS)) for c in full] + [(c, k, list(IN_PROCESS)) for c in rest]
    return chosen


def _frozen_record(cfg_id, attack):
    return read(FROZEN_CELLS / f'{cfg_id}_{attack}{SUFFIX}')


def select():
    if (DIR / 'selection.json').exists():
        raise SystemExit('selection exists; the sample is drawn once')
    by_cfg = frozen_population()
    keys = {}

    def fragment_count(cfg_id):
        # Cells copied from an exported group carry no key; any measured cell of the configuration does.
        if cfg_id not in keys:
            keys[cfg_id] = None
            for attack in sorted(by_cfg[cfg_id]):
                rec = _frozen_record(cfg_id, attack)
                if 'key' in rec:
                    key = tuple(map(tuple, rec['key']))
                    if CP.exact_cfg_id(key) != cfg_id:
                        raise ValueError(f'frozen cell {cfg_id} names another configuration')
                    keys[cfg_id] = key
                    break
        return None if keys[cfg_id] is None else len(keys[cfg_id][0])

    # only configurations that could enter the sample need their key read
    eligible = {c: a for c, a in by_cfg.items() if set(IN_PROCESS) <= a}
    sample = select_sample(eligible, fragment_count)
    items = []
    for cfg_id, k, attacks in sample:
        order, fe, strengths = keys[cfg_id]
        candidate = dict(order=list(order), fe_on=list(fe), fe=list(fe), s=dict(zip(order, strengths)))
        items.append(dict(cfg_id=cfg_id, fragments=k, attacks=attacks, candidate=candidate))
    atomic(DIR / 'selection.json', dict(policy=BRIDGE_POLICY, run_policy=POLICY, population_configs=len(by_cfg),
                                        eligible_configs=len(eligible), items=items,
                                        tolerances=dict(deterministic=TOL_DETERMINISTIC, stochastic=TOL_STOCHASTIC),
                                        stochastic=sorted(STOCHASTIC), deterministic=sorted(DETERMINISTIC)))
    print(json.dumps([(i['cfg_id'][:12], i['fragments'], len(i['attacks'])) for i in items]))


def submit(workers=6, light=False):
    selection = read(DIR / 'selection.json')
    paths = []
    from gpu import preflight
    for item in selection['items']:
        # The frozen driver judges its plan row (max_ms 1e20) before measuring and reads the candidate's
        # modeled latency and table PSNR; a frozen cell carries neither. Latency 0.0 and the frozen cell's
        # measured PSNR are used; neither can affect a measured cell.
        frozen = _frozen_record(item['cfg_id'], item['attacks'][0])
        candidate = dict(item['candidate'], ms=item['candidate'].get('ms', 0.0),
                         psnr_db=item['candidate'].get('psnr_db', frozen.get('psnr_db') or 0.0))
        task = dict(candidate=candidate, attacks=item['attacks'], cls=BRIDGE_CLS, run_cells_only=True)
        preflight(task)
        p = RUN / 'tasks' / ('bridge_' + task_id(task) + '.json')
        if not p.exists():
            atomic(p, task)
        paths.append(str(p))
    atomic(DIR / 'queue.json', paths)
    n = min(int(workers), len(paths))
    (RUN / 'logs').mkdir(exist_ok=True)
    gpu = ['-p', 'gpu', '--gres=gpu:l40s:1'] if light else \
        ['-p', 'gpu', '--gres=gpu:1', '--constraint=a100-80g']
    cmd = ['sbatch', '--parsable', *gpu, '-c', '6', '--mem=64G', '-t', '08:00:00',
           f'--array=0-{n - 1}%{n}', '-J', 'exp_mc_bridge', '-o', str(RUN / 'logs/bridge_%A_%a.log'),
           '--chdir=' + str(RUN), '--export=ALL', str(RUN / 'code/worker.sbatch'),
           'bridge_light' if light else 'bridge']
    job = subprocess.check_output(cmd, text=True, timeout=60).strip().split(';')[0]
    with (RUN / 'jobs.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind='bridge', job=job, workers=n)) + '\n')
    print('BRIDGE_SUBMITTED', job, n, flush=True)


def compare_cells(old, new, alphas_thresholds):
    """Per-bucket rate and mean differences of two cells of one configuration and attack."""
    rows = []
    for theta in alphas_thresholds:
        a, b = old['by_threshold'].get(theta), new['by_threshold'].get(theta)
        if a is None or b is None:
            return None
        da = [bool(x) for x in a['detected']]
        db = [bool(x) for x in b['detected']]
        if len(da) != len(db):
            return None
        rate_a, rate_b = sum(da) / len(da), sum(db) / len(db)
        mean_a = sum(a['accepted_score']) / len(da)
        mean_b = sum(b['accepted_score']) / len(db)
        rows.append(dict(threshold=theta, rate_old=rate_a, rate_new=rate_b, d_rate=rate_b - rate_a,
                         mean_old=mean_a, mean_new=mean_b, d_mean=mean_b - mean_a,
                         agreement=sum(x == y for x, y in zip(da, db)) / len(da)))
    return rows


def judge_attack(attack, rows):
    if not rows:
        return dict(approved=False, reason='no complete comparison')
    d_rate = [r['d_rate'] for r in rows]
    d_mean = [r['d_mean'] for r in rows]
    stats = dict(pairs_buckets=len(rows), max_abs_d_rate=max(map(abs, d_rate)), max_abs_d_mean=max(map(abs, d_mean)),
                 shift_rate=sum(d_rate) / len(d_rate), shift_mean=sum(d_mean) / len(d_mean),
                 min_agreement=min(r['agreement'] for r in rows))
    if attack in STOCHASTIC:
        t = TOL_STOCHASTIC
        ok = (abs(stats['shift_rate']) <= t['shift_rate'] and abs(stats['shift_mean']) <= t['shift_mean']
              and stats['max_abs_d_rate'] <= t['max_rate'] and stats['max_abs_d_mean'] <= t['max_mean'])
        kind = 'stochastic'
    else:
        t = TOL_DETERMINISTIC
        ok = stats['max_abs_d_rate'] <= t['max_rate'] and stats['max_abs_d_mean'] <= t['max_mean']
        kind = 'deterministic'
    return dict(approved=bool(ok), kind=kind, tolerances=t, **stats)


def decide():
    from unified_detector import threshold_count
    selection = read(DIR / 'selection.json')
    per_attack = collections.defaultdict(list)
    pairs, missing = [], []
    for item in selection['items']:
        c = item['candidate']
        key = CP.exact_cfg_key(c['order'], c['fe_on'], [c['s'][f] for f in c['order']])
        cfg = dict(order=c['order'], fe_on=c['fe_on'])
        thresholds = sorted({str(threshold_count(cfg, a)) for a in CP.REQUEST_ALPHAS}, key=int)
        for attack in item['attacks']:
            new = load_cell(c, attack, frozen=False)
            old = read(FROZEN_CELLS / cell_name(key, attack))
            if new is None:
                missing.append([item['cfg_id'], attack])
                continue
            rows = compare_cells(old['cell'], new['cell'], thresholds)
            if rows is None:
                missing.append([item['cfg_id'], attack])
                continue
            per_attack[attack] += rows
            pairs.append(dict(cfg_id=item['cfg_id'], fragments=item['fragments'], attack=attack,
                              psnr_old=old.get('psnr_db'), psnr_new=new.get('psnr_db'), buckets=rows))
    if missing:
        raise SystemExit(f'bridging measurements incomplete: {missing[:5]} ({len(missing)} missing)')
    verdicts = {a: judge_attack(a, per_attack.get(a, [])) for a in MAIN_ATTACKS}
    atomic(DIR / 'report.json', dict(policy=BRIDGE_POLICY, selection_sha256=hashlib.sha256(
        (DIR / 'selection.json').read_bytes()).hexdigest(), attacks=verdicts, pairs=pairs))
    atomic(APPROVAL, dict(policy=POLICY, bridge_policy=BRIDGE_POLICY, complete=True,
                          attacks={a: dict(approved=v['approved'], kind=v.get('kind')) for a, v in verdicts.items()}))
    print(json.dumps({a: v['approved'] for a, v in verdicts.items()}, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'select':
        select()
    elif cmd == 'submit':
        submit(sys.argv[2] if len(sys.argv) > 2 else 6, light='--light' in sys.argv[3:])
    elif cmd == 'decide':
        decide()
    else:
        raise SystemExit(__doc__)
