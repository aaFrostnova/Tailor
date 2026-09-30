"""SyncSeal (Meta FAIR, arXiv 2509.15208) as a plug-and-play LEARNED geometric-synchronization
front-end for the composite. Pure TorchScript inference (no syncseal package needed). It embeds a
dedicated sync watermark, then under geometric attack predicts the 4 image corners and perspective-
unwarps (rectifies) the frame — recovering rotation+scale+translation+crop+perspective in ONE pass.
crypto-verify stays the sole ACCEPT gate downstream, so FPR is unchanged (rectify is preprocessing).
Measured (syncseal_resync_test.py): crop75 detection 0.13->0.96, rot30 0.00->0.96, +43.5dB, no poison."""
import numpy as np, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.paths import SYNCSEAL_JIT as DEFAULT_JIT, require


def load_sync(jit_path: str = DEFAULT_JIT, dev: str = "cuda"):
    """Load the scripted SyncSeal model (embed/detect/unwarp)."""
    require(jit_path, "SyncSeal", "python tools/fetch_models.py --only syncseal prints where to "
            "put syncmodel.jit.pt", "TAILOR_SYNCSEAL_JIT")
    return torch.jit.load(jit_path).to(dev).eval()


def _pt(pil: Image.Image, dev: str) -> torch.Tensor:
    return to_tensor(pil).unsqueeze(0).to(dev)


def _pil(t: torch.Tensor) -> Image.Image:
    return Image.fromarray((t[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy())


def sync_embed(sync, pil: Image.Image, dev: str = "cuda") -> Image.Image:
    """Add the SyncSeal sync watermark on top of an image (same size). ~43.5 dB, no payload poison."""
    with torch.no_grad():
        ws = sync.embed(_pt(pil, dev))["imgs_w"]
    return _pil(ws)


def sync_rectify(sync, pil: Image.Image, dev: str = "cuda", size=(512, 512)):
    """Detect the 4 corners + perspective-unwarp -> (rectified PIL, sync presence score)."""
    with torch.no_grad():
        x = _pt(pil, dev)
        det = sync.detect(x)
        pts = det["preds_pts"]
        score = float(det["preds"][0, 0])
        r = sync.unwarp(x, pts, size)
    return _pil(r), score
