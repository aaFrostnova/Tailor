"""Shared SD-VAE latent helper for post-hoc latent-frequency watermarks.

Any latent-domain post-hoc watermark (phase modulation, ZoDiac-style
Fourier ring, etc.) needs to (a) VAE-encode an arbitrary image to the SD latent, modify
it, and (b) VAE-decode back to a watermarked image, then (c) re-encode a suspect image to
read the latent. This wraps a frozen SD AutoencoderKL with that contract.

The VAE is the detection anchor: detection re-encodes the suspect to the SAME latent the
watermark lives in. Regeneration / VAE round-trips operate in (or near) this same latent
manifold, which is why latent-domain marks survive them where pixel marks do not.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# Local SD checkpoints (VAE is the `vae` subfolder).
SD_PATHS = {
    "sd15": "/data/tailor/assets/model/stable-diffusion-v1-5",
    "sd21": "/data/tailor/assets/model/stable-diffusion-2-1-base",
    "sd14": "/data/tailor/assets/model/stable-diffusion-v1-4",
}


class LatentVAE:
    """Frozen SD VAE: image <-> latent (1,4,h/8,w/8), scaled by the VAE scaling factor."""

    def __init__(self, key: str = "sd15", device: str = "cuda", dtype=torch.float32):
        from diffusers import AutoencoderKL
        self.device = device
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(SD_PATHS[key], subfolder="vae",
                                                 torch_dtype=dtype).to(device).eval()
        self.scale = float(self.vae.config.scaling_factor)  # 0.18215 for SD-1.x

    @torch.no_grad()
    def encode(self, pil: Image.Image, sample: bool = False) -> torch.Tensor:
        """Image -> scaled latent. sample=False uses the posterior mean (deterministic)."""
        x = torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype) * 2 - 1
        dist = self.vae.encode(x).latent_dist
        lat = dist.sample() if sample else dist.mean
        return lat * self.scale

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, size=None) -> Image.Image:
        """Scaled latent -> image."""
        rec = self.vae.decode(latent.to(self.device, self.dtype) / self.scale).sample
        arr = ((rec.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        return out.resize(size, Image.LANCZOS) if size and out.size != size else out

    @torch.no_grad()
    def roundtrip(self, pil: Image.Image) -> Image.Image:
        """VAE encode->decode with no modification (fidelity floor of any latent mark)."""
        return self.decode(self.encode(pil), size=pil.size)
