"""CtrlRegen+ (RAVEN 'CtrlGen+', arXiv:2410.05470) batch attacker. RUN IN THE `ctrlregen` ENV.

Reads a dir of 512x512 PNGs and writes the regenerated (de-watermarked) versions to out_dir,
preserving filenames so the fingerprint-env decoder can match by name. One model load for the
whole batch. Must be launched with cwd = external/CtrlRegen (local imports).

  conda run -n ctrlregen python <repo>/experiments/attack/ctrlregen_batch.py \
      --in_dir ... --out_dir ... --step 0.5
"""
import argparse, glob, os, sys
import torch
import torchvision.transforms as T
from PIL import Image

CTRL = os.environ.get('TAILOR_PROJECT', '.') + "/external/CtrlRegen"
_ORIG_CWD = os.getcwd()                 # capture BEFORE chdir so relative I/O paths still resolve
sys.path.insert(0, CTRL)
os.chdir(CTRL)                          # CtrlRegen loads ckpts via repo-relative paths


def _resolve(p):
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_ORIG_CWD, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--step", type=float, default=0.5, help="regen strength in [0,1] (higher=more removal)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--step_list", type=str, default="", help="comma-sep strengths; sweep ALL in one model load -> {out_dir}_s{tag}")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    args.in_dir = _resolve(args.in_dir); args.out_dir = _resolve(args.out_dir)
    dev = "cuda"
    steps = [float(s) for s in args.step_list.split(",") if s.strip()] if args.step_list else [args.step]

    from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
    from custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
    from controlnet_aux import CannyDetector
    from transformers import AutoModel, AutoImageProcessor
    from utils import color_match

    resize512 = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512)])
    spatial = [ControlNetModel.from_pretrained("spatialnet_ckp/spatial_control_ckp_14000", torch_dtype=torch.float16)]
    pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V4.0_noVAE", controlnet=spatial, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.costum_load_ip_adapter("semanticnet_ckp", subfolder="models", weight_name="semantic_control_ckp_435000.bin")
    pipe.image_encoder = AutoModel.from_pretrained("facebook/dinov2-giant").to(dev, dtype=torch.float16)
    pipe.feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(1.0); pipe.set_progress_bar_config(disable=True); pipe.to(dev)
    canny = CannyDetector()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    for step in steps:
        odir = args.out_dir if not args.step_list else f"{args.out_dir}_s{str(step).replace('.', '')}"
        os.makedirs(odir, exist_ok=True)
        print(f"[ctrlregen] {len(files)} imgs, step={step} -> {odir}", flush=True)
        for k, fp in enumerate(files):
            # NON-512 SEMANTICS (pinned): resize-to-512 -> attack -> resize BACK to the native size.
            # The ControlNet checkpoints are 512-native, so a real attacker downsamples, regenerates and
            # restores the original geometry; without the resize-back the "attack" also silently became a
            # resolution change, which would confound any resolution-axis measurement.
            src = Image.open(fp).convert("RGB")
            native = src.size
            img = resize512(src)
            ctrl = canny(img, low_threshold=100, high_threshold=150)
            g = torch.manual_seed(args.seed)
            out = pipe("best quality, high quality",
                       negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                       image=[img], control_image=[ctrl], ip_adapter_image=[img], strength=step,
                       generator=g, num_inference_steps=args.steps, controlnet_conditioning_scale=1.0,
                       guidance_scale=2.0, control_guidance_start=0, control_guidance_end=1).images[0]
            out = color_match(img, out)
            if out.size != native:                      # restore native geometry (see semantics note above)
                out = out.resize(native, Image.BILINEAR)
            out.save(os.path.join(odir, os.path.basename(fp)))
            if (k + 1) % 16 == 0: print(f"  step={step} [{k+1}/{len(files)}]", flush=True)
    print("CTRLREGEN_BATCH_DONE")


if __name__ == "__main__":
    main()
