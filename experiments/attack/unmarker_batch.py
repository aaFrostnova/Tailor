"""Standalone UnMarker attack (RAVEN 'UnMarker', arXiv:2405.08363). RUN IN THE `unmarker` ENV.

UnMarker's attack is watermark-AGNOSTIC: its CW optimizer's loss is purely perceptual
(DeeplossVGG) + spectral (FFT/Mean), and the `evalu` arg is used ONLY for progress logging
(cw.py supports evalu=None). So we bypass the whole watermarker/evaluator stack (and the 30GB
download) by subclassing UnMark and skipping BaseAttack.__init__. We load our composite-
watermarked PNGs, run the 2-stage attack, and save the de-watermarked PNGs; the fingerprint
env then decodes them with our composite detector.

  conda run -n unmarker python ... --in_dir <embedded> --out_dir <attacked> --config attack_configs/Vine.yaml
"""
import argparse, glob, os, sys
import yaml
import torch
from PIL import Image
from torchvision import transforms

CTRL = os.environ.get('TAILOR_PROJECT', '.') + "/external/ai-watermark"
_ORIG = os.getcwd()
sys.path.insert(0, CTRL)
os.chdir(CTRL)  # loss_provider loads weights via repo-relative pretrained_models/... path

from modules.attack.unmark.unmark import UnMark  # noqa: E402


def _resolve(p):
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_ORIG, p))


class StandaloneUnMark(UnMark):
    """UnMark without the watermarker/evaluator (evalu=None) -> attacks arbitrary images."""

    def __init__(self, stage_selector, preprocess_args, stage1_args, stage2_args,
                 device="cuda", image_size=512):
        # --- bypass BaseAttack.__init__ (no init_watermarker, no FID/lpips eval) ---
        self.evaluator = None
        self.e = None
        self.image_size = image_size
        self.batch_size = 1
        self.device = device
        # --- replicate UnMark.__init__ body (transforms + two CW stages) ---
        stage_selector = ["preprocess", "stage1", "stage2"] if stage_selector is None else stage_selector
        all_args = [s if name in stage_selector else None
                    for name, s in zip(["preprocess", "stage1", "stage2"],
                                       (preprocess_args, stage1_args, stage2_args))]
        preprocess_args, stage1_args, stage2_args = all_args
        preprocess_args = {} if preprocess_args is None else preprocess_args
        crop_size = ((int(preprocess_args["crop_ratio"][0] * self.image_size),
                      int(preprocess_args["crop_ratio"][1] * self.image_size))
                     if preprocess_args.get("crop_ratio") is not None else None)
        crop_layer = transforms.CenterCrop(crop_size) if crop_size is not None else transforms.Lambda(lambda x: x)
        rescale_layer = (transforms.Resize((self.image_size, self.image_size), antialias=None)
                         if crop_size is not None else transforms.Lambda(lambda x: x))
        self.transforms = transforms.Compose([crop_layer, rescale_layer])
        self.stage1, self.stage1_thresh = self._load_stage(stage1_args, stage_name="high_freq")
        self.stage2, self.stage2_thresh = self._load_stage(stage2_args, stage_name="low_freq")

    def do_batch(self, x):
        # Memory-managed two-stage: detach + empty_cache between stages so stage1's
        # autograd graph/activations (~12GB) are freed before stage2 allocates its
        # FFT/VGG buffers -> fits in a 14.6GB GPU (default holds both -> OOM).
        stage0 = self.transforms(x)
        stage1 = self.stage1(stage0, stage0, ox=stage0, **self.stage1_thresh)
        if hasattr(stage1, "detach"): stage1 = stage1.detach()
        del stage0; torch.cuda.empty_cache()
        removed = self.stage2(stage1, stage1, ox=stage1, **self.stage2_thresh)
        del stage1; torch.cuda.empty_cache()
        return None, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="attack_configs/Vine.yaml")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start_idx", type=int, default=0, help="skip the first start_idx images (for SLURM-array chunking)")
    ap.add_argument("--batch", type=int, default=1, help="images per GPU forward (>1 = batched adversarial opt; needs a big GPU)")
    ap.add_argument("--max_iter_stage1", type=int, default=0, help="override (0=use config)")
    ap.add_argument("--max_iter_stage2", type=int, default=0)
    ap.add_argument("--no_preprocess", action="store_true", help="drop the 0.9 center-crop preprocess (pure spectral attack, alignment preserved)")
    ap.add_argument("--crop_ratio", type=float, default=0.0, help="override preprocess center-crop ratio (e.g. 0.95); 0=use config default")
    ap.add_argument("--lite_filter", action="store_true", help="drop 3 largest stage2 bilateral kernels to fit a 14.6GB GPU")
    args = ap.parse_args()
    in_dir = _resolve(args.in_dir); out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"

    conf = yaml.load(open(os.path.join(CTRL, args.config)), Loader=yaml.Loader)["UnMarker"]
    if args.max_iter_stage1: conf["stage1_args"]["max_iterations"] = args.max_iter_stage1
    if args.max_iter_stage2: conf["stage2_args"]["max_iterations"] = args.max_iter_stage2
    if args.lite_filter:
        # Drop the 3 largest bilateral-filter kernels from stage2's perceptual constraint
        # so the im2col/unfold buffers fit a 14.6GB GPU. These kernels are an IMPERCEPTIBILITY
        # constraint, so removing them only makes the attack freer/stronger (lower output SSIM,
        # which we measure) -> a fair-or-conservative adaptation, not a weakened attacker.
        ks = conf["stage2_args"]["filter_args"]["kernels"]
        big = {(47, 5), (33, 17), (17, 33)}
        newk = [tuple(k) for k in ks if tuple(k) not in big]
        conf["stage2_args"]["filter_args"]["kernels"] = newk
        # optimizer requires len(learning_rates) == len(kernels)+1; rebuild to match (lrs are all 0.01 + final 0.002)
        conf["stage2_args"]["optimizer_args"]["learning_rate"]["values"] = [0.01] * len(newk) + [0.002]
        print(f"[unmarker] lite_filter: stage2 kernels {len(ks)}->{len(newk)} (lrs->{len(newk)+1})", flush=True)

    ss = conf.get("stage_selector")
    if args.crop_ratio and args.crop_ratio > 0:
        conf.setdefault("preprocess_args", {})["crop_ratio"] = (args.crop_ratio, args.crop_ratio)
        if ss is not None and "preprocess" not in ss: ss = ["preprocess"] + list(ss)
        print(f"[unmarker] crop_ratio override -> {args.crop_ratio}", flush=True)
    if args.no_preprocess and ss:
        ss = [s for s in ss if s != "preprocess"]
        print(f"[unmarker] no_preprocess -> stage_selector={ss}", flush=True)
    atk = StandaloneUnMark(
        stage_selector=ss,
        preprocess_args=conf.get("preprocess_args"),
        stage1_args=conf.get("stage1_args"),
        stage2_args=conf.get("stage2_args"),
        device=dev, image_size=512)

    import gc
    files = sorted(glob.glob(os.path.join(in_dir, "*.png")))[args.start_idx:args.start_idx + args.n]
    B = max(1, args.batch)
    def _grp(lst):
        for i in range(0, len(lst), B): yield lst[i:i + B]
    print(f"[unmarker] {len(files)} imgs (start={args.start_idx}) from {in_dir} batch={B} (PHASED)", flush=True)
    to_t = transforms.ToTensor(); to_pil = transforms.ToPILImage()
    # --- phase 1: high-freq (stage1), batched; free stage1 before stage2 (peak = one stage) ---
    s1 = []
    done = 0
    for grp in _grp(files):
        xs = torch.stack([to_t(Image.open(fp).convert("RGB").resize((512, 512))) for fp in grp]).to(dev)
        st0 = atk.transforms(xs)
        out1 = atk.stage1(st0, st0, ox=st0, **atk.stage1_thresh)
        for j in range(out1.shape[0]): s1.append(out1[j:j+1].detach().cpu())
        del xs, st0, out1; torch.cuda.empty_cache(); done += len(grp)
        print(f"  stage1 [{done}/{len(files)}]", flush=True)
    del atk.stage1; atk.stage1 = None; gc.collect(); torch.cuda.empty_cache()
    # --- phase 2: low-freq (stage2) on the stage1 outputs, batched ---
    done = 0
    for gi in range(0, len(files), B):
        grp = files[gi:gi+B]; st1 = torch.cat([s1[gi+j] for j in range(len(grp))], 0).to(dev)
        removed = atk.stage2(st1, st1, ox=st1, **atk.stage2_thresh)
        for j, fp in enumerate(grp):
            to_pil(removed[j:j+1].detach().clamp(0, 1)[0].cpu()).save(os.path.join(out_dir, os.path.basename(fp)))
        del st1, removed; torch.cuda.empty_cache(); done += len(grp)
        print(f"  stage2 [{done}/{len(files)}]", flush=True)
    print("UNMARKER_BATCH_DONE")


if __name__ == "__main__":
    main()
