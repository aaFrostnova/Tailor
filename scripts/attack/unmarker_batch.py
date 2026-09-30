"""Apply UnMarker to a directory of images, one output per input.

UnMarker is a universal watermark-removal attack: two optimisation stages, one on the
high-frequency spectrum and one on the low, under a perceptual-similarity constraint, after
a 0.9 centre crop. Its dependencies conflict with this repository's environment, so it is
not importable here: this script runs under the interpreter of the environment UnMarker was
installed into, named by TAILOR_UNMARKER_PYTHON, and src/attacks.py invokes it through
CROSS_ENV.

    python scripts/attack/unmarker_batch.py --in_dir DIR --out_dir DIR \
        --config attack_configs/Vine.yaml --n 100 --start_idx 0 --batch 1

Both stages, the crop and every optimiser setting come from UnMarker's own config and its
own code; `--config` is a path inside its source tree and is read unchanged. Two things are
different from its `attack.py`, and both are because the images here are already
watermarked:

  * Its driver embeds a watermark itself, with its own copy of the scheme, and attacks
    that. This one attacks the images in `--in_dir`, which this repository watermarked.
  * Its driver therefore also reports detection rates, LPIPS and FID against that scheme.
    This one reports nothing: the detector that matters runs later, in this repository, and
    fidelity is measured there too. UnMarker's objective has no watermark term, so dropping
    the scheme changes the attack's own optimisation not at all.

Leaving the scheme out also cuts the install down. UnMarker's `systems/` package imports
all eleven schemes it can evaluate, so its own instructions build an environment for all of
them, and download the models of all of them; this driver needs the attack itself:

    pip install -r requirements.txt                  # in UnMarker's source tree
    pip install -e modules/attack/unmark             # its `special_loss` package

The attack's objective is a perceptual loss with weights of its own, which its
download_data_and_models.sh provides at pretrained_models/loss_provider/weights/. That
directory is still needed. The eleven schemes' own models are not.

An output that already exists is left alone, so an interrupted run resumes.
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from src.paths import UNMARKER_REPO, require    # noqa: E402

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


class _Unused:
    """Stands in for the FID metric, which this run never reads."""

    def to(self, _device):
        return self

    def reset(self):
        return None

    def update(self, *_a, **_k):
        return None


def _stand_in(image_size):
    """What UnMarker still wants from a watermarking scheme once it is not embedding one.

    BaseAttack reads `image_size` off the scheme and hands the scheme to each stage, which
    passes it to the progress bar and nowhere else (modules/attack/unmark/cw.py: the
    objective is the perceptual loss and the spectral distance). So the working resolution
    is the whole of it, and the progress bar's scheme is unset after construction.
    """
    import torch

    class StandIn:
        def __init__(self):
            self.image_size = image_size
            self.resizer = torch.nn.Identity()
            self.acceptance_thresh = None

    return StandIn()


def register_stand_in(image_size):
    """Put a `watermarkers` module in place before UnMarker's code imports one.

    modules/attack/base_attack.py does `from watermarkers import init_watermarker` while it
    is being imported, and that package pulls in all eleven watermarking schemes UnMarker
    supports, each with its own dependency tree, TensorFlow among them. Not one of them is
    wanted here, so registering the module first means the only dependencies to install are
    the attack's own.
    """
    import types

    module = types.ModuleType("watermarkers")
    module.init_watermarker = (
        lambda name, batch_size=1, device="cuda": _stand_in(image_size))
    sys.modules["watermarkers"] = module


def build(repo, config, image_size, batch, device):
    """UnMarker's attacker, with the watermarking scheme left out."""
    sys.path.insert(0, repo)
    register_stand_in(image_size)
    import yaml
    import modules.attack.base_attack as base

    # BaseAttack.__init__ resolves these in its own module globals when it runs, so
    # replacing them there is enough and every other line of upstream stays intact. The
    # similarity and distribution metrics would each load a network this run never reads,
    # one of them from a checkpoint only UnMarker's own data download provides.
    base.LpipsVGG = lambda device: None
    base.FrechetInceptionDistance = lambda: _Unused()

    from modules.attack import UnMark

    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    section = conf.get("UnMarker")
    assert section, f"{config} has no UnMarker section"

    attacker = UnMark("vine", input_dir=None, output_dir=None, device=device,
                      batch_size=batch, **section)
    for stage in (attacker.stage1, attacker.stage2):
        pbar = getattr(stage, "pbar", None)         # a disabled stage is a plain lambda
        if pbar is not None:
            pbar.evalu = None                       # no scheme, so no detection column
    return attacker


def load_batch(in_dir, names, size):
    """Our images as UnMarker's stages take them: (B, 3, size, size) in [0, 1]."""
    import torch
    from PIL import Image
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    images = []
    for name in names:
        image = Image.open(os.path.join(in_dir, name)).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.BICUBIC)
        images.append(to_tensor(image))
    return torch.stack(images)


def save_batch(removed, out_dir, names):
    import torch
    from torchvision import transforms

    to_pil = transforms.ToPILImage()
    for tensor, name in zip(removed.detach().cpu(), names):
        # The temp name keeps the real suffix, because PIL picks the format from it, and a
        # leading dot keeps it out of the scan in main().
        final = os.path.join(out_dir, name)
        stem, ext = os.path.splitext(name)
        partial = os.path.join(out_dir, f".{stem}.partial{ext}")
        to_pil(torch.clamp(tensor, 0.0, 1.0)).save(partial)
        os.replace(partial, final)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in_dir", required=True, help="images to attack")
    ap.add_argument("--out_dir", required=True, help="where the attacked images go, same names")
    ap.add_argument("--config", default="attack_configs/Vine.yaml",
                    help="UnMarker's own config, as a path inside its source tree")
    ap.add_argument("--n", type=int, default=0, help="how many images, 0 for all of them")
    ap.add_argument("--start_idx", type=int, default=0, help="where to start in the sorted names")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=512,
                    help="the resolution the stages work at; must be the images' own")
    ap.add_argument("--repo", default=UNMARKER_REPO, help="UnMarker's source tree")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    repo = require(a.repo, "UnMarker", "python tools/fetch_models.py --only unmarker",
                   "TAILOR_UNMARKER_REPO")
    in_dir, out_dir = os.path.abspath(a.in_dir), os.path.abspath(a.out_dir)

    names = sorted(f for f in os.listdir(in_dir)
                   if f.lower().endswith(EXTENSIONS) and not f.startswith("."))
    names = names[a.start_idx:] if a.n <= 0 else names[a.start_idx:a.start_idx + a.n]
    assert names, f"no images at {in_dir}[{a.start_idx}:]"
    os.makedirs(out_dir, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(out_dir, n))]
    print(f"unmarker {a.config}: {len(todo)} of {len(names)} images to do", flush=True)
    if not todo:
        return 0

    # Its configs and its own pretrained_models/ are resolved against its root, so the
    # attack runs from there. Our directories were made absolute above.
    cwd = os.getcwd()
    try:
        os.chdir(repo)
        config = a.config if os.path.isabs(a.config) else os.path.join(repo, a.config)
        require(config, "UnMarker's config", f"{a.config} is missing from {repo}")
        attacker = build(repo, config, a.image_size, a.batch, a.device)
        for start in range(0, len(todo), a.batch):
            chunk = todo[start:start + a.batch]
            _, removed = attacker.do_batch(load_batch(in_dir, chunk, a.image_size).to(a.device))
            save_batch(removed, out_dir, chunk)
            print(f"  {min(start + a.batch, len(todo))}/{len(todo)}", flush=True)
    finally:
        os.chdir(cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
