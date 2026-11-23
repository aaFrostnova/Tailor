"""W-Bench editing attacks: global editing (InstructPix2Pix) and local editing (SD inpainting).

Both are size-preserving (output resized back to input size). Used in addition to
the regeneration attack (already in attacks.py) to cover VINE's W-Bench editing axes.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_PIPES = {}

# A small fixed set of generic global-edit instructions (cycled by image index).
GLOBAL_EDIT_PROMPTS = [
    "make it look like a painting",
    "turn it into winter with snow",
    "make it a sunset scene",
    "give it autumn colors",
    "make it look vintage",
]


def _ip2p(device="cuda"):
    if "ip2p" not in _PIPES:
        from diffusers import (StableDiffusionInstructPix2PixPipeline,
                               EulerAncestralDiscreteScheduler)
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _PIPES["ip2p"] = pipe
    return _PIPES["ip2p"]


# Dedicated inpainting checkpoints (runwayml / stabilityai) are now gated or
# removed on the Hub, so local editing is implemented as "regenerate the masked
# region via img2img and composite it back" using a model we already have. This
# is a faithful local-editing attack: the masked box is fully resynthesised
# while the rest of the watermarked image is preserved.


def global_edit(pil, device="cuda", idx=0, seed=42):
    """InstructPix2Pix global edit (W-Bench global editing)."""
    pipe = _ip2p(device)
    W, H = pil.size
    ew, eh = (W // 8) * 8, (H // 8) * 8
    proc = pil.resize((ew, eh), Image.LANCZOS)
    prompt = GLOBAL_EDIT_PROMPTS[idx % len(GLOBAL_EDIT_PROMPTS)]
    g = torch.Generator(device).manual_seed(seed)
    out = pipe(prompt=prompt, image=proc, num_inference_steps=20,
               image_guidance_scale=1.5, guidance_scale=7.0, generator=g).images[0]
    return out.resize((W, H), Image.BILINEAR)


def _center_mask(W, H, frac=0.5):
    """White box (region to inpaint) covering frac of each dimension, centred."""
    m = np.zeros((H, W), np.uint8)
    bw, bh = int(W * frac), int(H * frac)
    l, t = (W - bw) // 2, (H - bh) // 2
    m[t:t + bh, l:l + bw] = 255
    return Image.fromarray(m, "L")


def local_edit(pil, device="cuda", frac=0.5, seed=42, model_key="sd21", strength=0.8):
    """Local editing: resynthesise a centred box via img2img, composite it back.

    The masked box is regenerated (diffusion edit) while the rest of the
    watermarked image is preserved. Tests whether a watermark survives when only
    part of the image is edited."""
    from wbench.attacks import _load_sd
    pipe = _load_sd(model_key, device)
    W, H = pil.size
    ew, eh = (W // 8) * 8, (H // 8) * 8
    proc = pil.resize((ew, eh), Image.LANCZOS)
    g = torch.Generator(device).manual_seed(seed)
    regen = pipe(prompt="", image=proc, strength=strength, num_inference_steps=50,
                 guidance_scale=1.0, generator=g).images[0].resize((W, H), Image.BILINEAR)
    mask = np.asarray(_center_mask(W, H, frac), np.float32)[..., None] / 255.0
    base = np.asarray(pil.convert("RGB"), np.float32)
    edit = np.asarray(regen.convert("RGB"), np.float32)
    out = (mask * edit + (1.0 - mask) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_editing(name, pil, device="cuda", idx=0):
    if name == "global_edit":
        return global_edit(pil, device, idx=idx)
    if name.startswith("local_edit"):
        frac = float(name.split("_")[2]) / 100.0 if name.count("_") >= 2 else 0.5
        return local_edit(pil, device, frac=frac)
    raise ValueError(f"unknown editing attack {name}")
