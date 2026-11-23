"""Margin backoff changes search allowances, never the user's four inputs or live gate."""
POLICY = 'initial_margin_unsat_backoff_v1'
STAGES = {'04': (0.04, 0.02), '02': (0.02, 0.01), '00': (0.0, 0.0)}
REQUEST_FIELDS = ('attacks', 'fpr', 'min_psnr', 'max_ms')


def schedule(origin_arm):
    assert origin_arm in ('04', '06')
    return ['04', '02', '00'] if origin_arm == '06' else ['02', '00']


def assert_same_request(a, b):
    assert a['i'] == b['i'] and all(a[f] == b[f] for f in REQUEST_FIELDS), 'fallback changed the request'
    assert 'min_bits' not in a and 'min_bits' not in b


def outcome(row):
    if row.get('open') or any(v['pending'] for v in row.get('walk', [])):
        return 'pending'
    if row['deployed_rank'] is not None:
        passed = [v for v in row['walk'] if v['pass'] and v['rank'] == row['deployed_rank']]
        assert len(passed) == 1 and not passed[0]['pending']
        return 'live_pass'
    if not row['candidates']:
        return 'no_candidate'
    assert row.get('walk') and all(not v['pass'] for v in row['walk'])
    return 'live_exhausted'


def make_attempt(row, stage, source):
    status = outcome(row)
    assert status != 'pending', 'cannot advance margin while live cells are missing'
    rate, mean = STAGES[stage]
    selected = None
    if status == 'live_pass':
        selected = next(v for v in row['walk'] if v['rank'] == row['deployed_rank'])
        selected = {k: selected[k] for k in ('rank', 'cfg_id', 'psnr_live', 'capacity_bits_estimate')}
    return dict(stage=stage, rate_margin=rate, mean_margin=mean, outcome=status,
                source=str(source), selected=selected)


def apply_attempt(state, attempt):
    assert state['status'] == 'pending_fallback'
    assert attempt['outcome'] in ('live_pass', 'no_candidate', 'live_exhausted'), 'incomplete fallback attempt'
    assert attempt['stage'] in schedule(state['origin_arm'])
    previous = [x['stage'] for x in state['attempts']]
    assert attempt['stage'] not in previous, 'duplicate fallback stage'
    assert schedule(state['origin_arm'])[len(previous)] == attempt['stage'], 'skipped a margin stage'
    state['attempts'].append(attempt)
    if attempt['outcome'] == 'live_pass':
        state['status'] = 'fallback_live_pass'
        state['selected'] = attempt['selected']
        state['selected_margins'] = [attempt['rate_margin'], attempt['mean_margin']]
    elif attempt['stage'] == '00':
        assert attempt['outcome'] in ('no_candidate', 'live_exhausted')
        state['status'] = ('zero_margin_model_unsat' if attempt['outcome'] == 'no_candidate'
                           else 'fallback_live_exhausted')
    return state


def strict_optimize_checks():
    """Z3 unknown is an execution failure, never an UNSAT or optimum certificate."""
    import z3
    original = z3.Optimize.check
    if getattr(original, '_rejects_unknown', False):
        return
    def check(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result == z3.unknown:
            raise RuntimeError('Z3 returned unknown: ' + str(self.reason_unknown()))
        return result
    check._rejects_unknown = True
    z3.Optimize.check = check
