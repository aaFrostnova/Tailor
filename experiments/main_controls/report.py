"""Periodic summary of the main controls; partial rates are never labeled final."""
import collections
import statistics
from common import POLICY, RUN, atomic

SCENARIOS = ('S1', 'S2', 'S3', 'S4', 'S5')


def summarize(rows, variant):
    group = [r for r in rows if r['variant'] == variant]
    accepted = [r for r in group if r['status'] == 'accepted']
    scenarios = {}
    for s in SCENARIOS:
        subset = [r for r in group if r['cls'] == 'C' + s[1:]]
        passed = [r for r in subset if r['status'] == 'accepted']
        scenarios[s] = dict(n=len(subset), accepted=len(passed),
                            percent=100 * len(passed) / len(subset) if subset else None,
                            psnr=statistics.mean(r['verdict']['psnr_live'] for r in passed) if passed else None)
    origins = collections.Counter('+'.join(r['walk'][-1].get('origins') or ['none']) for r in accepted)
    stages = collections.Counter(f"{r['stage']}/rank{r['rank']}" for r in accepted)
    complete = all(not r['status'].startswith('pending') for r in group)
    return dict(n=len(group), accepted=len(accepted), percent=100 * len(accepted) / len(group) if group else None,
                macro_percent=(statistics.mean(v['percent'] for v in scenarios.values())
                               if all(v['n'] for v in scenarios.values()) else None),
                psnr=statistics.mean(r['verdict']['psnr_live'] for r in accepted) if accepted else None,
                status_counts=dict(collections.Counter(r['status'] for r in group)), scenarios=scenarios,
                accepted_cell_origins=dict(origins), accepted_stage_rank=dict(stages), complete=complete,
                label='final' if complete else 'partial (pending requests remain)')


def publish(rows, status):
    variants = sorted({r['variant'] for r in rows})
    atomic(RUN / 'summary.json', dict(policy=POLICY, progress=status,
                                      variants={v: summarize(rows, v) for v in variants}))
