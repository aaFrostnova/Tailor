"""Everything this repository needs and does not contain.

No model, no image and no measurement output is redistributed here: the three fragments,
the learned geometric front-end, the baseline and the attack code all come from their own
projects, and the measurement images are rebuilt. This is the single place that decides
where they are looked for, so a fresh checkout needs no edit to the code:
`tools/fetch_models.py` puts each one at the default below, and one environment variable
moves any one of them elsewhere.

Source trees that have to be importable go under $TAILOR_MODELS (default `external/`) and
loose checkpoints beside them. The diffusion attack models default to their Hugging Face
identifiers instead, because `from_pretrained` resolves and caches an identifier on first
use -- a local directory is an override there, not a requirement.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(var, default):
    """An environment variable if it is set to something, else the bundled default."""
    return os.environ.get(var) or default


MODELS = _env("TAILOR_MODELS", os.path.join(REPO_ROOT, "external"))

# ------------------------------------------------------------------------- roots
# PROJECT is this checkout, which the campaigns put on sys.path to import each other.
# WORKSPACE is where a campaign writes: its measurements, its shards and the measured
# cell cache. Nothing under it is an input, so it is safe to point at scratch space.
PROJECT = _env("TAILOR_PROJECT", REPO_ROOT)
WORKSPACE = _env("TAILOR_WORKSPACE", os.path.join(REPO_ROOT, "workspace"))
# The interpreter a campaign launches its GPU tasks with, this one unless told otherwise.
PYTHON = _env("TAILOR_PYTHON", sys.executable)
# One campaign's run tree. prepare_campaign.py freezes this checkout's code and the request
# sets it enumerated under it, so a run stays reproducible against the code that produced
# it, and the scripts under measurement/ are the source that gets frozen, not the copy.
RUN_ROOT = _env("TAILOR_RUN_ROOT", os.path.join(WORKSPACE, "campaign"))

# ---------------------------------------------------------------------- fragments
# VINE: the source tree defines VINE_Turbo and CustomConvNeXt; the weights are the two
# Hugging Face repositories those classes load, which they fetch themselves.
VINE_REPO = _env("TAILOR_VINE_REPO", os.path.join(MODELS, "VINE"))
VINE_HF = "Shilin-LU/VINE-{variant}-{part}"          # variant R or B, part Enc or Dec
# TrustMark: the pip package carries its own weights and fetches them on first use, so
# there is nothing to place.
# VideoSeal: the source tree resolves its model card relative to its own root.
VIDEOSEAL_REPO = _env("TAILOR_VIDEOSEAL_REPO", os.path.join(MODELS, "videoseal"))

# ------------------------------------------------------- learned geometric front-end
# A TorchScript export, used only by the geometric stage that rectifies before decoding.
SYNCSEAL_JIT = _env("TAILOR_SYNCSEAL_JIT", os.path.join(MODELS, "syncseal", "syncmodel.jit.pt"))

# ---------------------------------------------------------------------- baseline
MASKWM_REPO = _env("TAILOR_MASKWM_REPO", os.path.join(MODELS, "MaskWM"))
MASKWM_CKPT = _env("TAILOR_MASKWM_CKPT", os.path.join(MASKWM_REPO, "checkpoints", "D_128bits.pth"))

# ------------------------------------------------------------------ attack models
# The regeneration attack is Zhao et al.'s implementation: `regen_pipe` and `wmattacker`
# are imported from this source tree, which carries no weights of its own.
WMATTACKER_REPO = _env("TAILOR_WMATTACKER_REPO", os.path.join(MODELS, "WatermarkAttacker"))


# The diffusion model the regeneration attack rebuilds an image with. stabilityai gates
# its repositories: accept the terms on the model page and `hf auth login`, or set this to
# a local directory. The neural-compression attacks need no model here; compressai fetches
# its own. `SD` stays a mapping so a campaign can add a variant without touching callers.
SD = {"sd21": _env("TAILOR_SD21", "stabilityai/stable-diffusion-2-1")}

# CtrlRegen and UnMarker each need dependencies that conflict with this environment, so
# they are run out of process. There is no default: the interpreter of the environment
# each one was installed into has to be named.
CROSS_ENV_PYTHON = {
    "ctrlregen": _env("TAILOR_CTRLREGEN_PYTHON", ""),
    "unmarker":  _env("TAILOR_UNMARKER_PYTHON", ""),
}

# ---------------------------------------------------------------------- image pool
# The measurement images. Rebuilt by pipeline/build_wm_dataset.py, not redistributed.
POOL = _env("TAILOR_POOL", os.path.join(WORKSPACE, "pool"))


def require(path, what, how, var=None):
    """Return `path`, or say what is missing, where it was looked for, and how to get it."""
    if path and os.path.exists(path):
        return path
    where = f"looked for it at {path}" if path else "no location is set for it"
    move = f"\n  {var} moves it somewhere else." if var else ""
    raise FileNotFoundError(f"{what} is not installed: {where}.\n  {how}{move}")
