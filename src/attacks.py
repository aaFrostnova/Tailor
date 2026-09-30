"""The single definition of every attack, shared by the measurement campaigns and the reported matrix.

An attack that is implemented twice is two attacks. This module exists because they had drifted: the
in-process surrogate campaigns restated the operators locally, and the restatements disagreed with the
ones the reported matrix uses -- crop read its fraction as an area rather than a side, rotation filled
the corners with black instead of reflecting, and the photometric and noise attacks were substantially
milder (brightness measured 23.9 dB against the cover where the reported operator measures 11.5 dB).
The solver was therefore planning against a weaker threat than the evaluation scored it on, and nothing
in either program could notice, because both were internally consistent.

Everything here is PIL-in / PIL-out so a campaign can call it in-process, while `attack_file` keeps the
path-based form the evaluation harness uses. The two share one implementation, so they cannot diverge.
"""
import io, os, tempfile
import numpy as np
from PIL import Image

from src.paths import SD as _SD, CROSS_ENV_PYTHON as _CROSS_PY, PROJECT as _PROJECT

# ---------------------------------------------------------------- geometry (pure PIL, no deps)
# CONVENTION. Two conventions were in use and had to be reconciled; the one adopted here is the
# AREA-fraction crop and the default-fill rotation. Both are stated explicitly because the alternative
# reading of each name denotes a materially different attack, and the earlier divergence between the
# measurement campaigns and the evaluation harness was exactly this ambiguity going unnoticed.
def center_crop_area(img, area_frac):
    """Keep a centred `area_frac` of the AREA and resize back to the original canvas.

    The side length is therefore sqrt(area_frac): "crop75" keeps three quarters of the picture, not
    three quarters of each edge. Reading the same number as a side fraction gives a substantially
    harsher attack (0.75 per side is 56% of the area), which is the discrepancy that previously made
    the surrogate and the reported matrix disagree about what "crop75" meant.
    """
    W, H = img.size; s = area_frac ** 0.5
    cw, ch = int(W * s), int(H * s)
    l, t = (W - cw) // 2, (H - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)

# kept under the old name so callers that mean "a fraction of each side" cannot silently get the other
def center_crop_resize(img, frac):
    """Deprecated alias. `frac` is interpreted as an AREA fraction, matching center_crop_area."""
    return center_crop_area(img, frac)

def rotate_fill(img, deg):
    """Rotate in place with the default fill, leaving the corners black.

    Note for interpretation: the black corners are lost content, but they are also a strong
    localisation cue, and a decoder with a geometric front-end can use them to recover the transform.
    A reflection-padded rotation destroys less of the image yet is harder to invert, because the
    original frame's corners end up outside the returned view. Numbers measured under this operator
    should be read as the easier of the two for a front-end that predicts corners.
    """
    W, H = img.size
    return img.rotate(deg, resample=Image.BICUBIC, expand=False).resize((W, H), Image.BICUBIC)

def rotate_reflect(img, deg):
    """Reflection-padded rotation. Retained for the ablation that contrasts the two conventions; it is
    NOT what GEO['rot9'] applies."""
    W, H = img.size; arr = np.array(img); pad = max(W, H) // 2
    refl = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))

def resize_down_up(img, mid):
    W, H = img.size
    return img.resize((mid, mid), Image.BICUBIC).resize((W, H), Image.BICUBIC)

def crop_then_jpeg(img, frac=0.75, quality=25):
    out = center_crop_area(img.convert("RGB"), frac)          # same area convention as crop75
    buf = io.BytesIO(); out.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

def border_crop(img, frac=0.20):
    """Translation crop: drop a border of `frac` of the side from the top and left, then push the
    surviving content to the origin, leaving the vacated right/bottom black.

    This is a DIFFERENT operator from center_crop_area, and the difference is the whole point: a
    center crop keeps the picture centred and only rescales it, so a mark that survives rescaling
    survives it, whereas this one TRANSLATES the content off its embedding grid. Plain TrustMark
    holds the centre crops and breaks here, which is the column a spatially redundant embedding is
    for. Frame size is preserved so the operator composes with the rest of the suite.
    """
    W, H = img.size
    d = int(round(min(W, H) * float(frac)))
    reg = img.crop((d, d, W, H))
    out = Image.new("RGB", (W, H))
    out.paste(reg, (0, 0))
    return out


GEO = {
    "crop75": lambda im: center_crop_area(im, 0.75),          # 75% of the AREA (side 0.866)
    "crop50": lambda im: center_crop_area(im, 0.50),          # 50% of the AREA (side 0.707)
    "rot9":   lambda im: rotate_fill(im, 9.0),                # default fill, corners go black
    "rs256":  lambda im: resize_down_up(im, 256),
    "hflip":  lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "crop_jpeg": crop_then_jpeg,
    "border20": lambda im: border_crop(im, 0.20),         # TRANSLATION crop, not a centre crop
}

# ---------------------------------------------------------------- signal family (WAVES numerics)
# Parameters are the ones the reported matrix runs; they live here so a campaign cannot pick others.
SIGNAL_PARAMS = {"jpeg": dict(quality=25), "blur": dict(ksize=5, sigma=1),
                 "noise": dict(std=0.05), "bright": dict(factor=0.2),
                 "contrast": dict(factor=0.2), "bm3d": dict()}
_SIG_CACHE = {}

def _signal_attackers():
    if not _SIG_CACHE:
        from wmattacker import (GaussianBlurAttacker, GaussianNoiseAttacker, JPEGAttacker,
                                BrightnessAttacker, ContrastAttacker, BM3DAttacker)
        p = SIGNAL_PARAMS
        _SIG_CACHE.update({
            "jpeg":     JPEGAttacker(quality=p["jpeg"]["quality"]),
            "blur":     GaussianBlurAttacker(p["blur"]["ksize"], p["blur"]["sigma"]),
            "noise":    GaussianNoiseAttacker(std=p["noise"]["std"]),
            "bright":   BrightnessAttacker(p["bright"]["factor"]),
            "contrast": ContrastAttacker(p["contrast"]["factor"]),
            "bm3d":     BM3DAttacker(),
        })
    return _SIG_CACHE

ALIASES = {"jpeg25": "jpeg", "vaeB": "vae_b", "vaeC": "vae_c"}

def attack_pil(name, img, dev="cuda", _vae_cache={}):
    """Apply one attack to a PIL image and return a PIL image. Geometry is pure PIL; the signal family
    routes through the same attacker objects the harness uses, via a temp file, so there is exactly one
    parameterisation. VAE is lazy so a geometry-only campaign needs no compressai."""
    name = ALIASES.get(name, name)
    if name == "clean":
        return img.convert("RGB")
    if name in GEO:
        out = GEO[name](img.convert("RGB"))
        return out if out.size == img.size else out.resize(img.size, Image.BICUBIC)
    if name in ("vae_b", "vae_c"):
        if name not in _vae_cache:
            from wmattacker import VAEWMAttacker
            mn = {"vae_b": "bmshj2018-hyperprior", "vae_c": "cheng2020-anchor"}[name]
            _vae_cache[name] = VAEWMAttacker(mn, quality=3, metric="mse", device=dev)
        att = _vae_cache[name]
    else:
        att = _signal_attackers().get(name)
        if att is None:
            raise ValueError(f"unknown attack: {name}")
    d = tempfile.mkdtemp()
    try:
        ip, op = os.path.join(d, "i.png"), os.path.join(d, "o.png")
        img.convert("RGB").save(ip)
        att.attack([ip], [op])
        return Image.open(op).convert("RGB")
    finally:
        for f in ("i.png", "o.png"):
            try: os.remove(os.path.join(d, f))
            except OSError: pass
        try: os.rmdir(d)
        except OSError: pass


# ---------------------------------------------------------------- diffusion family (in-process)
# The mild regeneration and its repeats. These were built independently in the harness and in each
# regeneration campaign; the noise step in particular decides how destructive the attack is, so it is
# declared once here.
SD21_PATH = _SD["sd21"]
REGEN_PARAMS = dict(noise_step=60, batch_size=1)
RINSE_PASSES = {"regen": 1, "rinse2x": 2, "rinse4x": 4}
_REGEN = {}

def regen_attacker(dev="cuda", sd_path=SD21_PATH):
    """The single mild-regeneration attacker. Loaded lazily: a campaign that touches no diffusion
    attack must not pay for Stable Diffusion."""
    if "att" not in _REGEN:
        import torch
        from regen_pipe import ReSDPipeline
        from wmattacker import DiffWMAttacker
        from diffusers import DPMSolverMultistepScheduler
        pipe = ReSDPipeline.from_pretrained(sd_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _REGEN["att"] = DiffWMAttacker(pipe.to(dev), **REGEN_PARAMS)
    return _REGEN["att"]

def regen_pil(img, passes=1, dev="cuda", workdir=None):
    """Apply the regeneration `passes` times in sequence -- the repo's rinse definition."""
    import tempfile, shutil
    att = regen_attacker(dev)
    d = workdir or tempfile.mkdtemp()
    try:
        cur = os.path.join(d, "r_in.png"); img.convert("RGB").save(cur)
        for k in range(passes):
            nxt = os.path.join(d, f"r_{k}.png"); att.attack([cur], [nxt]); cur = nxt
        return Image.open(cur).convert("RGB")
    finally:
        if workdir is None:
            shutil.rmtree(d, ignore_errors=True)

# ---------------------------------------------------------------- cross-environment attacks
# CtrlRegen and UnMarker need their own conda environments, so they cannot be called in-process.
# What CAN be centralised is how they are invoked, which is what drifted between run scripts.
# `script` is absolute: the driver is in this checkout, while the interpreter belongs to
# the attack's own environment, so neither side shares a working directory with the other.
CROSS_ENV = {
    "ctrlregen": {"env": _CROSS_PY["ctrlregen"],
                  "script": os.path.join(_PROJECT, "scripts/attack/ctrlregen_batch.py"),
                  "default_step": 0.7, "sweep_steps": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "unmarker":  {"env": _CROSS_PY["unmarker"],
                  "script": os.path.join(_PROJECT, "scripts/attack/unmarker_batch.py"),
                  "config": "attack_configs/Vine.yaml"},
}

# The attacks a campaign may run in-process. Anything outside this set has to be staged through
# CROSS_ENV and is reported as such rather than silently skipped.
IN_PROCESS = set(GEO) | set(SIGNAL_PARAMS) | {"vae_b", "vae_c"} | set(RINSE_PASSES) | {"clean"}

def attack_pil_any(name, img, dev="cuda"):
    """Dispatch over every in-process attack, including the diffusion family."""
    name = ALIASES.get(name, name)
    if name in RINSE_PASSES:
        return regen_pil(img, passes=RINSE_PASSES[name], dev=dev)
    return attack_pil(name, img, dev=dev)
