"""Prepare paired four-input requests without retaining payload-derived floors."""
import collections
import hashlib
import json
from pathlib import Path
from campaign import ROOT, INPUTS, SC, CLASSES, ARMS, FIELDS, atomic, boot, checkpoint, verify_checkpoint
from capacity_protocol import PROTOCOL, REQUEST_FIELDS, solver_scenario


def main():
    assert not (ROOT / 'prepared.json').exists(), 'do not overwrite prepared requests'
    boot('04')
    import watermark_smt_v2 as W
    import solver_eval_continuous as SEC
    from class_defs import classes, sample_class
    old_root = SC / 'topk_full_20260910'
    changes, reuse = {}, {}
    for ci, k in enumerate(CLASSES):
        old = json.loads((old_root / 'inputs' / (k + '.json')).read_text())['records']
        scenarios0 = sample_class(classes(SEC.sg.attacks)[ci], ci, 2000, W.beta_from_fpr, capacity_output=True)
        records, scenarios = [], []
        changed = collections.Counter()
        for i, (before, s) in enumerate(zip(old, scenarios0)):
            assert before['i'] == i and before['attacks'] == s['attacks'] and before['fpr'] == s['fpr'], (k, i, 'paired request drift')
            assert 'min_bits' not in s
            r = {f: s[f] for f in REQUEST_FIELDS}
            r.update(i=i, cls=k, protocol=PROTOCOL, min_ba=s['min_ba'],
                     k=s['_k'], stress=s['_stress'], stress_kind=s['_stress_kind'],
                     cr=s['_cr'], unm=s['_unm'], optional=s['_optional'],
                     ceiling_fpr=s['_ceiling_fpr'], hardest=s['_hardest'])
            sc = dict(r, allow_resync=True, allow_nested=True)
            assert solver_scenario(r)['min_bits'] == 0
            records.append(r); scenarios.append(sc)
            for f in ('min_psnr', 'max_ms', 'stress'):
                changed[f] += before[f] != r[f]
        atomic(INPUTS / (k + '.json'), {'protocol': PROTOCOL, 'records': records, 'scenarios': scenarios})
        changes[k] = dict(changed)
        for arm in ARMS:
            count = 0
            # Only a certified solve with zero capacity requirement and identical
            # remaining constraints may be reused. Positive min_bits never qualifies.
            for p in sorted((old_root / 'checkpoints' / (k + '_' + arm)).glob('q*.json')):
                d = json.loads(p.read_text()); r = records[d['i']]
                if d['min_bits'] != 0 or any(d[f] != r[f] for f in FIELDS):
                    continue
                d['mean_margin'] = d.pop('cap_margin')
                d.pop('min_bits'); d.update(protocol=PROTOCOL, stress=r['stress'],
                    stress_kind=r['stress_kind'], margin_run=None, reused_enumeration=str(p))
                verify_checkpoint(d, r, arm)
                atomic(checkpoint(k, arm, r['i']), d); count += 1
            reuse[k + '_' + arm] = count
    batches = [[[k, arm, i] for i in range(start, start + 10)]
               for start in range(0, 2000, 10) for k in reversed(CLASSES) for arm in ARMS]
    coverage = [tuple(x) for batch in batches for x in batch]
    assert len(coverage) == len(set(coverage)) == 20000
    atomic(ROOT / 'plans/enumeration.json', batches)
    atomic(ROOT / 'prepared.json', dict(protocol=PROTOCOL, request_inputs=list(REQUEST_FIELDS),
        n_original_requests=10000, n_arm_request_evaluations=20000, rate_margins=[.04, .06],
        n_images=100, unmarker_images=30, live_acceptance_rate_min=.9,
        capacity_role='output_only', psnr_role='maximum fidelity subject to four inputs',
        paired_with=str(old_root), changed_from_payload_run=changes, reused_enumerations=reuse,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS.glob('*.json')}))
    print(json.dumps({'protocol': PROTOCOL, 'changes': changes, 'reused': reuse}, indent=2))


if __name__ == '__main__':
    main()
