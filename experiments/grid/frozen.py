"""Load the frozen TAILOR solver code for the grid baseline C2 (CPU only, nothing is embedded or decoded).

The frozen modules are byte-identical copies under RUN/root (experiments/main_controls/common.py). boot() runs
campaign.boot(arm) (RUN/root/fixed/code/campaign.py:66-90) exactly as the main controls do, with one step
skipped: unified_detector.install() (campaign.py:78-79) imports eval_matrix (torch, PIL, the GPU decoder) only
to patch OursComposite.decode, and the grid never decodes. boot(decoder=True) runs campaign.boot verbatim.
Every module the grid reads is checked to come from the relocated root.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

DEFAULT_RUN = os.environ.get('TAILOR_RUNS', 'runs') + '/main_controls_20261109'
MAIN_CONTROLS = Path(__file__).resolve().parents[1] / 'main_controls'
ARM = '06'
# experiments/protocol.md section 7: surrogate_canonical.json sha256 55309914...b99e5, 8,305,481 bytes.
SURROGATE_SHA256 = '5530991458b71acf90bbdea860d741a3a72bd26ea3d2ea99f81948d7950b99e5'
_STATE = {}


def run_dir():
    return Path(os.environ.setdefault('TAILOR_MAIN_CONTROLS_RUN', DEFAULT_RUN)).resolve()


def available():
    return (run_dir() / 'root/fixed/code/campaign.py').exists()


def _origin(module, directory):
    if Path(module.__file__).resolve().parent != Path(directory):
        raise ImportError(f'{module.__name__} loaded from {module.__file__}, expected {directory}')


def boot(arm=ARM, decoder=False):
    """The frozen solver modules as a namespace; idempotent within a process."""
    if 'ns' in _STATE:
        if _STATE['ns'].arm != arm:
            raise RuntimeError('frozen code already booted for another arm')
        return _STATE['ns']
    run_dir()
    if str(MAIN_CONTROLS) not in sys.path:
        sys.path.insert(0, str(MAIN_CONTROLS))
    import common
    import unified_detector
    _origin(unified_detector, common.ROOT / 'code')
    original = unified_detector.install
    if not decoder:
        unified_detector.install = lambda target=None: None
    try:
        common.C.boot(arm)
    finally:
        unified_detector.install = original
    import capacity_protocol
    import composite_prediction
    import margin_fallback_policy
    import surrogate_model
    import watermark_smt_topk
    import watermark_smt_v2
    for module in (watermark_smt_v2, surrogate_model, watermark_smt_topk, capacity_protocol, margin_fallback_policy):
        _origin(module, common.ROOT / 'fixed/code')
    _origin(composite_prediction, common.ROOT / 'code')
    expected_floor = common.ROOT / 'fixed/inputs/clean_minimum_strength.json'
    if Path(watermark_smt_v2.CLEAN_FLOOR_JSON).resolve() != expected_floor:
        raise ImportError(f'clean floors read from {watermark_smt_v2.CLEAN_FLOOR_JSON}, expected {expected_floor}')
    ns = types.SimpleNamespace(arm=arm, common=common, C=common.C, ROOT=common.ROOT, RUN=common.RUN,
                               W=watermark_smt_v2, SM=surrogate_model, T=watermark_smt_topk,
                               CP=capacity_protocol, MFP=margin_fallback_policy, CPD=composite_prediction,
                               UD=unified_detector)
    _STATE['ns'] = ns
    return ns


def surrogate_path(ns=None):
    return (ns or boot()).ROOT / 'fixed/inputs/surrogate_canonical.json'


def load_surrogate(ns=None):
    """(Surrogate, sha256) of the frozen table, loaded as common.install_variant loads it (common.py:203)."""
    ns = ns or boot()
    data = surrogate_path(ns).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SURROGATE_SHA256:
        raise ValueError(f'surrogate_canonical.json changed: {digest}')
    return ns.SM.Surrogate.from_dict(json.loads(data)), digest


def requests_path():
    return run_dir() / 'inputs/requests.json'
