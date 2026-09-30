"""Apply CtrlRegen to a directory of images, one output per input.

CtrlRegen regenerates an image from noise under semantic and spatial control, which
removes a watermark while keeping the picture. Its dependencies conflict with this
repository's environment, so it is not importable here: this script runs under the
interpreter of the environment CtrlRegen was installed into, named by
TAILOR_CTRLREGEN_PYTHON, and src/attacks.py invokes it through CROSS_ENV.

The pipeline is built exactly as CtrlRegen's own ctrlregen_plus_demo.ipynb builds it, and
`--step`, `--steps` and `--seed` are that notebook's `step`, `num_inference_steps` and
`seed`. What this script adds is the batch: the notebook attacks one image it has opened
itself, and a campaign needs a directory of images already watermarked by this repository.

    python scripts/attack/ctrlregen_batch.py --in_dir DIR --out_dir DIR \
        --step 0.7 --steps 50 --seed 1

An output that already exists is left alone, so an interrupted run resumes.
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from src.paths import CTRLREGEN_CKPT, CTRLREGEN_REPO, require   # noqa: E402

# The demo's constants. The two checkpoints are the trained controls, from CtrlRegen's own
# weight repository; the other three are public models it builds on.
SPATIAL_CONTROL = "spatialnet_ckp/spatial_control_ckp_14000"
SEMANTIC_CONTROL = "semanticnet_ckp"
SEMANTIC_WEIGHTS = "semantic_control_ckp_435000.bin"
DIFFUSION_MODEL = "SG161222/Realistic_Vision_V4.0_noVAE"
IMAGE_ENCODER = "facebook/dinov2-giant"
VAE = "stabilityai/sd-vae-ft-mse"
PROMPT = "best quality, high quality"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"
EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def build(repo, ckpt, device):
    """CtrlRegen's pipeline, loaded once for the whole directory."""
    sys.path.insert(0, repo)                    # custom_i2i_pipeline, custom_ip_adapter, utils
    cwd = os.getcwd()
    try:
        os.chdir(repo)
        import torch
        from controlnet_aux import CannyDetector
        from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
        from custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
        from utils import color_match

        spatial = ControlNetModel.from_pretrained(os.path.join(ckpt, SPATIAL_CONTROL),
                                                  torch_dtype=torch.float16)
        pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            DIFFUSION_MODEL, controlnet=[spatial], torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
        pipe.costum_load_ip_adapter(os.path.join(ckpt, SEMANTIC_CONTROL), subfolder="models",
                                    weight_name=SEMANTIC_WEIGHTS)

        from transformers import AutoImageProcessor, AutoModel
        pipe.image_encoder = AutoModel.from_pretrained(IMAGE_ENCODER).to(device, dtype=torch.float16)
        pipe.feature_extractor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER)
        pipe.vae = AutoencoderKL.from_pretrained(VAE).to(dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_ip_adapter_scale(1.0)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        return pipe, CannyDetector(), color_match, torch
    finally:
        os.chdir(cwd)


def regenerate(pipe, canny, color_match, torch, image, step, steps, seed):
    """One image, as the demo's ctrl_regen_plus does it.

    The generator is seeded per image, which is the demo's behaviour: every image in a run
    starts from the same noise, so a run is reproducible from --seed alone.
    """
    from torchvision import transforms
    to_512 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ])
    generator = torch.manual_seed(seed)
    image = to_512(image)
    control = canny(image, low_threshold=100, high_threshold=150)
    out = pipe(PROMPT,
               negative_prompt=NEGATIVE_PROMPT,
               image=[image],
               control_image=[control],          # spatial condition
               ip_adapter_image=[image],         # semantic condition
               strength=step,
               generator=generator,
               num_inference_steps=steps,
               controlnet_conditioning_scale=1.0,
               guidance_scale=2.0,
               control_guidance_start=0,
               control_guidance_end=1,
               ).images[0]
    return color_match(image, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in_dir", required=True, help="images to attack")
    ap.add_argument("--out_dir", required=True, help="where the attacked images go, same names")
    ap.add_argument("--step", type=float, required=True,
                    help="removal strength in [0, 1]: how far towards clean noise the latent "
                         "is taken before the controlled denoising")
    ap.add_argument("--steps", type=int, default=50, help="denoising steps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repo", default=CTRLREGEN_REPO, help="CtrlRegen's source tree")
    ap.add_argument("--ckpt", default=CTRLREGEN_CKPT, help="CtrlRegen's two trained controls")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    how = "python tools/fetch_models.py --only ctrlregen"
    require(a.repo, "CtrlRegen", how, "TAILOR_CTRLREGEN_REPO")
    require(os.path.join(a.ckpt, SPATIAL_CONTROL), "CtrlRegen's spatial control network",
            how, "TAILOR_CTRLREGEN_CKPT")
    assert 0.0 <= a.step <= 1.0, f"--step is a fraction, got {a.step}"

    from PIL import Image
    names = sorted(f for f in os.listdir(a.in_dir)
                   if f.lower().endswith(EXTENSIONS) and not f.startswith("."))
    assert names, f"no images in {a.in_dir}"
    os.makedirs(a.out_dir, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(a.out_dir, n))]
    print(f"ctrlregen step={a.step} steps={a.steps} seed={a.seed}: "
          f"{len(todo)} of {len(names)} images to do", flush=True)
    if not todo:
        return 0

    pipe, canny, color_match, torch = build(a.repo, a.ckpt, a.device)
    for i, name in enumerate(todo, 1):
        image = Image.open(os.path.join(a.in_dir, name)).convert("RGB")
        out = regenerate(pipe, canny, color_match, torch, image, a.step, a.steps, a.seed)
        # Write beside the target and rename, so an interrupted run never leaves a
        # half-written file that the resume check would then skip. The temp name keeps the
        # real suffix, because PIL picks the format from it, and a leading dot keeps it out
        # of the scan above.
        final = os.path.join(a.out_dir, name)
        stem, ext = os.path.splitext(name)
        partial = os.path.join(a.out_dir, f".{stem}.partial{ext}")
        out.save(partial)
        os.replace(partial, final)
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
