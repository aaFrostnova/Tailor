"""Expose the original experiment's exact embed/read/cell functions for new attacks."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT, WORKSPACE
SC = Path(WORKSPACE)
BASE = SC / 'topk_capacity_output'
CF = Path(PROJECT)


def load():
    import numpy as np
    from PIL import Image
    for path in (CF, CF / 'scripts', CF / 'scripts/defense', SC, BASE / 'code'):
        sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('_measurement_campaign', BASE / 'code/campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    campaign.boot('04')
    import watermark_smt_v2 as W
    from eval_matrix import OursComposite
    from src.soft_bch import decode_and_verify
    path = BASE / 'code/certify_full_frozen.py'
    source = path.read_text()
    names = {'composite_for', 'read', 'cell_of'}
    funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs} == names
    ns = dict(np=np, Image=Image, W=W, OursComposite=OursComposite, dev='cuda',
              decode_and_verify=decode_and_verify,
              FKEY={'VINE': 'vine', 'TrustMark': 'trustmark', 'VideoSeal': 'videoseal'})
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), ns)
    ns['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    return ns
