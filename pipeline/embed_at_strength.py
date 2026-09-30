"""Embed one fragment at one strength into N covers -- the staging half of a cross-environment run."""
import sys, os, glob, argparse
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC
for p in (CF, f"{CF}/solver"):
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["vine", "trustmark", "videoseal"])
ap.add_argument("--strength", type=float, required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--start_idx", type=int, default=0)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
if a.method == "vine":
    from src.vine_crypto_wrapper import VineCryptoWrapper
    F = VineCryptoWrapper(KEY, "vine", NB, dev, variant="R")
elif a.method == "trustmark":
    from src.trustmark_fragment import TrustMarkFragment
    F = TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev)
else:
    from src.videoseal_fragment import VideoSealFragment
    F = VideoSealFragment(KEY, "videoseal", NB, device=dev)

files = _pool_sample(a.n, offset=a.start_idx)   # across all five sources, in the evaluation set's proportions
for i, fp in enumerate(files):
    cov = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
    t = np.random.RandomState(a.start_idx + i).randint(0, 2, NB).astype(np.uint8)
    emb = F.embed_with_target(cov, t, strength=a.strength)
    if emb.size != (512, 512): emb = emb.resize((512, 512))
    emb.save(os.path.join(a.out_dir, f"i{a.start_idx + i:05d}.png"))
print(f"embedded {len(files)} @ {a.method} s={a.strength} -> {a.out_dir}  "
      f"sources={composition_of(files)}", flush=True)
