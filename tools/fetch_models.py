"""Put the external models where src/paths.py looks for them.

Nothing in this repository is a model. Solving requests needs none of them, because the
measured database in `inputs/` is what the solver reads; embedding an image, running an
attack or re-running the live calibration needs the models below.

    python tools/fetch_models.py                 # the three fragments
    python tools/fetch_models.py --check         # report what is present, fetch nothing
    python tools/fetch_models.py --only vine sd  # one or more components

Two components the script cannot fetch, so it says where to put them instead of guessing:
SyncSeal, whose TorchScript export the geometric front-end loads, and MaskWM, which is
only a baseline. TAILOR_MODELS moves the whole tree; the per-component variables in
src/paths.py move one of them.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import paths as M                                    # noqa: E402

GIT = {
    "vine":      ("https://github.com/Shilin-LU/VINE.git", M.VINE_REPO),
    "videoseal": ("https://github.com/facebookresearch/videoseal.git", M.VIDEOSEAL_REPO),
    # the regeneration attack: `regen_pipe` and `wmattacker`, no weights of its own
    "regen":     ("https://github.com/XuandongZhao/WatermarkAttacker.git", M.WMATTACKER_REPO),
}
MANUAL = {
    "syncseal": (M.SYNCSEAL_JIT, "TAILOR_SYNCSEAL_JIT",
                 "the TorchScript export `syncmodel.jit.pt` from the SyncSeal project "
                 "(arXiv 2509.15208). Only the geometric front-end that rectifies before "
                 "decoding uses it; every other stage runs without it."),
    "maskwm":    (M.MASKWM_CKPT, "TAILOR_MASKWM_CKPT",
                  "`D_128bits.pth` from the MaskWM project, under a checkpoints/ directory "
                  "in its source tree. It is a baseline, so nothing in the method needs it."),
}
ALL = ["trustmark", "vine", "videoseal", "regen", "syncseal", "maskwm", "sd"]


def say(component, status, detail=""):
    print(f"  {component:<10} {status:<8} {detail}")


def clone(url, dest, check):
    """Clone `url` to `dest` unless it is already there."""
    if os.path.isdir(os.path.join(dest, ".git")) or os.path.isdir(dest) and os.listdir(dest):
        return "present", dest
    if check:
        return "missing", f"would clone {url}"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, dest])
    return ("cloned", dest) if r.returncode == 0 else ("failed", f"git clone {url}")


def warm(component, fn, check):
    """Load a model once so its weights land in the local cache."""
    if check:
        return "skipped", "loads its own weights on first use"
    try:
        fn()
        return "cached", "weights in the local cache"
    except Exception as e:                                          # noqa: BLE001
        return "failed", f"{type(e).__name__}: {e}".replace("\n", " ")[:160]


def do_trustmark(check):
    def load():
        from trustmark import TrustMark
        TrustMark(verbose=False, model_type="B", use_ECC=False, secret_len=100)
    try:
        import trustmark                                            # noqa: F401
    except ImportError:
        return "missing", "pip install trustmark"
    return warm("trustmark", load, check)


def do_vine(check):
    status, detail = clone(*GIT["vine"], check=check)
    if status in ("missing", "failed"):
        return status, detail

    def load():
        sys.path[:0] = [M.VINE_REPO, os.path.join(M.VINE_REPO, "vine", "src")]
        from vine_turbo import VINE_Turbo
        from stega_encoder_decoder import CustomConvNeXt
        for part, cls in (("Enc", VINE_Turbo), ("Dec", CustomConvNeXt)):
            cls.from_pretrained(M.VINE_HF.format(variant="R", part=part))
    w, wd = warm("vine", load, check)
    return w, f"{M.VINE_REPO}; {wd}" if w == "cached" else wd


def do_videoseal(check):
    status, detail = clone(*GIT["videoseal"], check=check)
    if status in ("missing", "failed"):
        return status, detail

    def load():
        sys.path.insert(0, M.VIDEOSEAL_REPO)
        import videoseal
        cwd = os.getcwd()
        try:                                     # the model card resolves against its root
            os.chdir(M.VIDEOSEAL_REPO)
            videoseal.load("videoseal")
        finally:
            os.chdir(cwd)
    w, wd = warm("videoseal", load, check)
    return w, f"{M.VIDEOSEAL_REPO}; {wd}" if w == "cached" else wd


def do_manual(component, _check):
    path, var, what = MANUAL[component]
    if os.path.exists(path):
        return "present", path
    return "manual", f"put {what}\n{'':<21}-> {path}   ({var} to move it)"


def do_sd(check):
    """The diffusion attack models: the largest download here, and only the attacks use them."""
    if check:
        return "skipped", " ".join(sorted(set(M.SD.values())))
    try:
        from diffusers import AutoencoderKL, StableDiffusionPipeline
    except ImportError:
        return "missing", "pip install diffusers"
    # the regeneration attack loads a whole pipeline; the VAE attacks load only a `vae` subfolder
    jobs = [(M.SD["sd21"], lambda r: StableDiffusionPipeline.from_pretrained(r))]
    jobs += [(r, lambda r: AutoencoderKL.from_pretrained(r, subfolder="vae"))
             for r in sorted({M.SD[k] for k in ("sd21_base", "sd15", "sd14")})]
    bad = []
    for repo, load in jobs:
        try:
            load(repo)
        except Exception as e:                                      # noqa: BLE001
            bad.append(f"{repo} ({type(e).__name__})")
    if bad:
        return "failed", ("; ".join(bad) + ".  stabilityai gates its repositories: accept "
                          "the terms on the model page and `hf auth login`, or point "
                          "TAILOR_SD21 at a local directory.")
    return "cached", "weights in the local cache"


HANDLERS = {"trustmark": do_trustmark, "vine": do_vine, "videoseal": do_videoseal,
            "regen": lambda c: clone(*GIT["regen"], check=c),
            "syncseal": lambda c: do_manual("syncseal", c),
            "maskwm": lambda c: do_manual("maskwm", c), "sd": do_sd}
DEFAULT = ["trustmark", "vine", "videoseal", "regen"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=ALL, metavar="COMPONENT",
                    help=f"one or more of: {', '.join(ALL)} (default: {', '.join(DEFAULT)})")
    ap.add_argument("--all", action="store_true",
                    help="every component, including the diffusion attack models")
    ap.add_argument("--check", action="store_true",
                    help="report what is present and fetch nothing")
    a = ap.parse_args()
    want = a.only or (ALL if a.all else DEFAULT)

    print(f"\nmodels under {M.MODELS}\n")
    out = {c: HANDLERS[c](a.check) for c in want}
    for c, (status, detail) in out.items():
        say(c, status, detail)
    missing = [c for c, (s, _) in out.items() if s in ("missing", "failed")]
    manual = [c for c, (s, _) in out.items() if s == "manual"]
    print()
    if manual:
        print(f"place by hand: {', '.join(manual)}")
    if missing:
        print(f"not installed: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
