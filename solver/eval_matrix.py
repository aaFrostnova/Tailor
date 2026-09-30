"""Unified watermark eval matrix: ALL methods (Ours composite + post-hoc baselines) through
ONE canonical RAVEN attack suite.

The attack machinery (attackers + attack_one + GEO) is copied VERBATIM from
benchmark_composite_defense.py so Ours and every baseline see IDENTICAL attacks/strengths.
Per method: embed -> quality (PSNR/SSIM) -> for each attack: attack -> decode -> bit-acc +
detection (TPR@1%FPR). One method is loaded at a time (memory-lean). Writes one JSON per
method into --out_dir; make_matrix_table.py merges them into the exp_plan tables.

Ours = VideoSeal -> TrustMark -> VINE fused (crypto BCH(100,37), soft-LLR fuse, dual detect:
fused crypto-verify @2^-37 OR fused zero-bit >= 1%-FPR tau). Baselines: native bits, detect =
bit-acc >= per-method 1%-FPR binomial tau.
"""
import argparse, glob, os, sys, json, tempfile, io, math, time
import numpy as np, torch
from PIL import Image
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as REPO, SD as _SD, WMATTACKER_REPO, WORKSPACE
sys.path.insert(0, os.path.join(REPO, "solver"))
sys.path.insert(0, WMATTACKER_REPO)

SD21 = _SD["sd21"]
KEY = b"v5_key_encoder_master"
ATTACKS = ["clean", "jpeg", "blur", "noise", "bright", "contrast", "bm3d", "regen", "rinse2x", "rinse4x", "vae_b", "vae_c"]
# Listed rather than derived so the reported column ORDER is stable, but checked against the shared
# definition below: an attack added to src.attacks.GEO and not added here would be measured by the
# surrogate and never by the matrix, and the two would quietly describe different suites.
GEO_ATTACKS = ["crop75", "crop50", "rot9", "rs256", "hflip", "crop_jpeg", "border20"]
ALL_ATTACKS = ATTACKS + GEO_ATTACKS

# ---------------- attack layer ----------------
# Defined once, in src/attacks.py, and imported by the measurement campaigns as well. They used to be
# restated in each program, and the restatements had drifted: the same names denoted different
# operators in the table the solver reads and in the table we report.
from src.attacks import (GEO, center_crop_resize as _center_crop_resize,
                         rotate_reflect as _rotate_reflect, resize_down_up as _resize_down_up,
                         crop_then_jpeg as _crop_then_jpeg_pil, SIGNAL_PARAMS)
assert set(GEO_ATTACKS) == set(GEO), (
    f"the matrix's geometric suite and the shared definitions disagree: "
    f"only in GEO_ATTACKS {sorted(set(GEO_ATTACKS) - set(GEO))}, "
    f"only in src.attacks.GEO {sorted(set(GEO) - set(GEO_ATTACKS))}")

def _crop_then_jpeg(ip, op, frac=0.75, quality=25):
    _crop_then_jpeg_pil(Image.open(ip).convert("RGB"), frac=frac, quality=quality).save(op)

def build_attack_one(dev):
    # cheap signal attackers eagerly; regen (SD-2-1 on /project) + VAE (compressai) are LAZY so a
    # fast-attacks-only run never loads them (no /project dependency, no SD download).
    from wmattacker import (GaussianBlurAttacker, GaussianNoiseAttacker, JPEGAttacker,
                            BrightnessAttacker, ContrastAttacker, BM3DAttacker)
    P = SIGNAL_PARAMS                                  # one parameterisation, shared with campaigns
    sig = {"jpeg": JPEGAttacker(quality=P["jpeg"]["quality"]),
           "blur": GaussianBlurAttacker(P["blur"]["ksize"], P["blur"]["sigma"]),
           "noise": GaussianNoiseAttacker(std=P["noise"]["std"]),
           "bright": BrightnessAttacker(P["bright"]["factor"]),
           "contrast": ContrastAttacker(P["contrast"]["factor"]), "bm3d": BM3DAttacker()}
    lazy = {}
    def _regen():
        if "regen" not in lazy:
            from regen_pipe import ReSDPipeline
            from wmattacker import DiffWMAttacker
            from diffusers import DPMSolverMultistepScheduler
            pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
            lazy["regen"] = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
        return lazy["regen"]
    def _vae(k):
        if k not in lazy:
            from wmattacker import VAEWMAttacker
            mn = {"vae_b": "bmshj2018-hyperprior", "vae_c": "cheng2020-anchor"}[k]
            lazy[k] = VAEWMAttacker(mn, quality=3, metric="mse", device=dev)
        return lazy[k]

    def attack_one(a, ip, op):
        if a == "clean": Image.open(ip).save(op); return
        if a in sig: sig[a].attack([ip], [op]); return
        if a in ("vae_b", "vae_c"): _vae(a).attack([ip], [op]); return
        if a == "regen": _regen().attack([ip], [op]); return
        if a in ("rinse2x", "rinse4x"):
            k = 2 if a == "rinse2x" else 4; cur = ip; regen = _regen()
            for r in range(k):
                nxt = op.replace(".png", f"_r{r}.png"); regen.attack([cur], [nxt]); cur = nxt
            Image.open(cur).save(op); return
        if a in GEO:
            out = GEO[a](Image.open(ip).convert("RGB"))
            if out.size != (512, 512): out = out.resize((512, 512))
            out.save(op); return
        if a == "crop_jpeg":
            _crop_then_jpeg(ip, op); return
        raise ValueError(a)
    return attack_one

# ---------------- in-env advanced attacks (editing / img2video), PIL -> list[PIL] ----------------
_ADV_PIPES = {}
ADV_ATTACKS = {"editing", "img2video"}
def is_adv(a):
    return a in ADV_ATTACKS or a.startswith("local_edit")

def apply_adv(a, pil, idx, dev="cuda"):
    """In-env advanced attack. Returns a LIST of attacked PILs (14 frames for img2video)."""
    if a == "editing":                                   # W-Bench global editing (InstructPix2Pix)
        from wbench.editing import global_edit
        return [global_edit(pil, dev, idx=idx)]
    if a.startswith("local_edit"):                       # local_edit<frac> (masked img2img)
        from wbench.editing import local_edit
        frac = float(a.split("_")[2]) / 100.0 if a.count("_") >= 2 else 0.5
        return [local_edit(pil, dev, frac=frac)]
    if a == "img2video":                                 # Stable Video Diffusion, 14 frames
        if "svd" not in _ADV_PIPES:
            from diffusers import StableVideoDiffusionPipeline
            p = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16").to(dev)
            p.set_progress_bar_config(disable=True); _ADV_PIPES["svd"] = p
        g = torch.Generator(dev).manual_seed(42 + idx)
        cond = pil.resize((1024, 576))                   # SVD-XT native
        frames = _ADV_PIPES["svd"](cond, num_frames=14, decode_chunk_size=4,
                                   motion_bucket_id=127, noise_aug_strength=0.02, generator=g).frames[0]
        return [f.convert("RGB").resize((512, 512)) for f in frames]
    raise ValueError(a)

# ---------------- quality ----------------
def psnr(a, b):  # a,b float [0,1] HxWx3
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)

def ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as sk
        return float(sk(a, b, channel_axis=2, data_range=1.0))
    except Exception:
        # global fallback (Wang SSIM constants, whole-image single window)
        mu1, mu2 = a.mean(), b.mean(); v1, v2 = a.var(), b.var(); cov = ((a - mu1) * (b - mu2)).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (v1 + v2 + c2)))

# ---------------- Ours composite ----------------
class OursComposite:
    name = "Ours"; n_bits = 100
    ORDER = ["videoseal", "trustmark", "vine"]  # deployed embed order (VINE on top)
    SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}

    DEFAULT_STRENGTH = {"vine": 1.0, "trustmark": 1.0, "videoseal": 1.0}

    def __init__(self, dev, tm_variant="B", geo=False, vine_variant="R", config=None):
        """`config` is the solver's output c=(S, pi, s, r, nu), as
        {"frags": [...], "order": [...], "strengths": {f: float}, "resync": bool, "nested": bool}.
        When it is None the class falls back to the pinned reference point, which is a fixed operating
        point for the M1 comparison and NOT what the method returns for a request."""
        from src.shortened_bch import ShortenedBCH
        from src.vine_crypto_wrapper import VineCryptoWrapper
        from src.trustmark_fragment import TrustMarkFragment
        from src.videoseal_fragment import VideoSealFragment
        self.sb = ShortenedBCH(); self.tau = float(binom.ppf(0.99, self.sb.n, 0.5) + 1) / self.sb.n
        self.dev = dev
        self.frag = {
            "vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=self.sb.n, device=dev, variant=vine_variant),
            "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=self.sb.n, model_type=tm_variant, device=dev),
            "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=self.sb.n, device=dev),
        }
        self.name = f"Ours{'-B' if vine_variant == 'B' else ''}"   # VINE-B variant = higher-PSNR fragment option
        cfg = config or {}
        # The request's false-positive budget for the zero-bit presence tests (the keyed tests have their own,
        # fixed, 2^-37). The solver reasons about the request's budget and the certification re-thresholds
        # every recorded read at it; until 2026-09-09 the deployed decoder itself always ran at 1 percent.
        # `config["alpha"]` (or decode(..., alpha=)) sets it; the default keeps the 1 percent behaviour.
        self.alpha = float(cfg.get("alpha", 0.01))
        self.order = list(cfg.get("order") or cfg.get("frags") or self.ORDER)
        for f in self.order:
            if f not in self.frag: raise ValueError(f"unknown fragment in config: {f}")
        self.strength = dict(self.DEFAULT_STRENGTH); self.strength.update(cfg.get("strengths") or {})
        # Each geometric front-end is a separate decision. They used to collapse into one `geo`
        # flag, which made the cascade run all four of its stages whenever either of the two
        # booleans was set: asking for resync silently also paid for the VINE scale search, and
        # asking for the nested ring silently also paid for SyncSeal and the blind angle sweep.
        # The solver charges these separately, so they have to be separately switchable.
        self.nested = bool(cfg.get("nested", geo))            # embed-side: nested VINE ring
        self.resync = bool(cfg.get("resync", geo))            # embed-side: SyncSeal mark + rectify
        self.fe_scale = bool(cfg.get("scale_search", self.nested))   # decode-side: ring scale search
        self.fe_angle = bool(cfg.get("angle_sweep", self.resync))    # decode-side: blind angle probe
        # Spatially redundant TrustMark: the same crypto codeword in every cell of a 2x2 grid, with a
        # sliding-window crypto-verify at decode. The centre crops in the suite rescale the picture
        # and leave it centred, which plain TrustMark survives; a TRANSLATION crop moves the content
        # off the embedding grid, and this is the front-end for that column.
        self.fe_tile = bool(cfg.get("tile", False))                  # both-side: tiled embed + window search
        self.geo = self.nested or self.resync or self.fe_scale or self.fe_angle or self.fe_tile
        if config:
            self.name = "Ours[" + "+".join(self.order) + \
                        ("|R" if self.resync else "") + ("|N" if self.nested else "") + \
                        ("|S" if self.fe_scale else "") + ("|A" if self.fe_angle else "") + \
                        ("|T" if self.fe_tile else "") + "]"
        elif geo:
            self.name += "+geo"
        if self.geo:
            from composite_external_eval import scale_resid, nested_vine_embed, rot
            from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
            self._scale_resid, self._nested, self._rot = scale_resid, nested_vine_embed, rot
            self._sync_embed, self._sync_rectify = sync_embed, sync_rectify
            # The SyncSeal model is only needed by the resync stage. A config that asks only for the
            # blind angle sweep is a decode-side front-end with no embed-side mark and no model.
            self._sync = load_sync(dev=dev) if self.resync else None
            if self.fe_tile:
                from src.tiled_trustmark import TiledTrustMark
                self._tiled = TiledTrustMark(self.frag["trustmark"], self.sb)
            else:
                self._tiled = None
            self._head3 = None                              # learned head retired; see decode()
            self._rr, self._rs, self._vss, self._bc, self._bg = 180.0, 3.0, 0.005, 10.0, 0.60  # geo search params

    def secret_for(self, idx):
        image_id = f"img_{idx:05d}"; tx = self.sb.encode(_id_payload(image_id, self.sb.data_bits))
        return (image_id, tx)

    def embed(self, pil, idx):
        from src.vine_crypto_wrapper import apply_crypto
        image_id, tx = self.secret_for(idx)
        img = pil.resize((512, 512)) if pil.size != (512, 512) else pil
        for name in self.order:                               # subset AND order come from the config
            m = self.frag[name]; perm, M = m.get_perm_M(image_id)
            w = apply_crypto(tx, perm, M); s = float(self.strength.get(name, 1.0))
            if name == "vine" and self.nested:                # nested ring is a VINE-side front-end
                img = self._nested(m, img, w, scales=(1.0, 0.75, 0.5), strength=s)
            elif name == "trustmark" and self.fe_tile:        # tiled ring is a TrustMark-side front-end
                img = self._tiled.embed(img, w, strength=s)
            else:
                img = m.embed_with_target(img, w, strength=s)
            if img.size != (512, 512): img = img.resize((512, 512))
        if self.resync:                                       # sync mark goes on top, once
            img = self._sync_embed(self._sync, img, self.dev)
        return img, (image_id, tx)

    def _frag_llr(self, name, pil, iid):
        from src.soft_fusion import method_soft_to_codeword_llr
        kind, getter = self.SPEC[name]; m = self.frag[name]; perm, M = m.get_perm_M(iid)
        return method_soft_to_codeword_llr(getattr(m, getter)(pil), perm, M, kind=kind, n_codeword=self.sb.n)

    def _cv(self, rl, iid):
        from src.soft_bch import decode_and_verify
        return bool(decode_and_verify(rl, iid, codec=self.sb)["detected"])

    def geo_cascade(self, att, iid, tx, return_view=False):
        """Crypto-verify-gated geometric search over four separately selectable stages:
        S1/S2 SyncSeal rectify + residual tilt (self.resync), S3 VINE ring scale search
        (self.fe_scale), S4 blind angle probe (self.fe_angle). A stage the config did not ask
        for does not run, so its latency is not paid and its effect is not credited."""
        import numpy as _np
        present = [f for f in ("vine", "trustmark", "videoseal") if f in self.order]
        if not present: return (False, None) if return_view else False
        if self.resync:                                       # S1+S2: SyncSeal rectify, then residual tilt
            rect, _ = self._sync_rectify(self._sync, att, self.dev)
            for name in present:
                if self._cv(self._frag_llr(name, rect, iid), iid):
                    return (True, rect) if return_view else True
            for d in (-3.0, 3.0, -6.0, 6.0):                  # residual tilt after rectification
                rimg = self._rot(rect, d)
                for name in present:                      # every present fragment, not a chosen one
                    if self._cv(self._frag_llr(name, rimg, iid), iid):
                        return (True, rimg) if return_view else True
        if self.fe_scale and "vine" in present:               # S3: VINE ring scale search
            for f in _np.arange(0.34, 1.0001, self._vss):
                if f >= 0.999: view = att
                else:
                    s = int(round(512 * float(f))); o = (512 - s) // 2; view = att.crop((o, o, o + s, o + s))
                if self._cv(self._frag_llr("vine", view, iid), iid):
                    return (True, view) if return_view else True
        # Angle candidates come from a fragment-independent probe (src/angle_probe.py). The sweep used
        # to be driven by one payload fragment and to abort on that fragment's bit accuracy, which asks
        # a mark the rotation may already have destroyed how to undo the rotation: a configuration
        # whose fragments are rotation-fragile made the sweep inert, so whether rotation was
        # recoverable depended on which fragments the request happened to select. The probe reads the
        # image instead, proposes an ordering, and every candidate is still admitted only by the keyed
        # verification -- so a longer list costs time, never false accepts.
        if self.fe_tile and "trustmark" in present:           # S5: sliding-window crypto-verify
            perm, M = self.frag["trustmark"].get_perm_M(iid)
            ok, win = self._tiled.crop_recover(att, iid, perm, M, return_view=True)
            if ok: return (True, win) if return_view else True
        if self.fe_angle:                                     # S4: fragment-independent blind angle sweep
            from src.angle_probe import candidate_angles
            for d in candidate_angles(att, search=self._rr, step=self._bc, refine=self._rs):
                rimg = self._rot(att, float(d))
                for name in present:
                    if self._cv(self._frag_llr(name, rimg, iid), iid):
                        return (True, rimg) if return_view else True
        return (False, None) if return_view else False

    def decode_no_cascade(self, att, secret, alpha=None):
        """The same decode with the geometric cascade suppressed.

        The surrogate stores each front-end as an EFFECT: what the cascade adds over the same decoder
        without it. That subtraction is only the cascade's own contribution if both sides run the
        identical decoder on the identical image, which is what this pairing gives -- it differs from
        `decode` in exactly one branch."""
        saved = self.geo
        self.geo = False
        try:
            return self.decode(att, secret, alpha=alpha)
        finally:
            self.geo = saved

    def decode(self, att, secret, alpha=None):
        """`alpha`: the request's presence budget (default: the configured self.alpha, 1 percent unless the
        configuration carries one). It sets the zero-bit thresholds and thereby when the cascade fires."""
        from src.soft_fusion import fuse_llrs, llr_to_bits
        from src.soft_bch import decode_and_verify
        alpha = self.alpha if alpha is None else float(alpha)
        image_id, tx = secret; aligned = {}; bestpath = False
        for name in self.order:
            a = self._frag_llr(name, att, image_id); aligned[name] = a
            if decode_and_verify(a, image_id, codec=self.sb)["detected"]: bestpath = True
        # Equal-weight fusion for every configuration. A learned head over the three fragments' LLRs
        # used to take over here when all three were selected with a front-end on; it fired on 49 of
        # 8,575 solved requests and was measured at +0.4 points, and it was the pipeline's only learned
        # component. Removed so that every number in the evaluation comes from the same decoder.
        fused = fuse_llrs(aligned, weights=None, n_codeword=self.sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        fver = bool(decode_and_verify(fused, image_id, codec=self.sb)["detected"])
        # Presence is best-path as well: one live fragment fires the zero-bit test even when a dead
        # partner dilutes the fused value below threshold (crop-then-JPEG read 0.63 on VideoSeal and
        # 0.58 fused, and the fused-only test fired on 23.5% of images the solver had covered). The
        # per-fragment and fused tests share the 1% budget, so their thresholds rise with the count.
        per_ba = {n: float(np.mean(llr_to_bits(a) == tx)) for n, a in aligned.items()}
        det = fver or bestpath or presence_detected(per_ba, fba, len(aligned), alpha=alpha)
        if self.geo and not det:                              # geometric cascade fires only on primary miss
            ok, view = self.geo_cascade(att, image_id, tx, return_view=True)
            if ok:
                det = True
                # Report the accuracy of the view the decoder ACCEPTED. Left on `att`, this number
                # describes an alignment the decoder did not use: the crop50 cell read 0.55 -- a
                # miss -- while the deployed decoder was accepting the image, so bit accuracy and
                # detection disagreed on every cell the cascade rescued. It also makes the
                # front-end's effect measurable in the same unit as the base curves, which is what
                # the coverage clause compares against beta.
                rl = {n: self._frag_llr(n, view, image_id) for n in self.order}
                fba = float(np.mean(llr_to_bits(fuse_llrs(rl, weights=None, n_codeword=self.sb.n)) == tx))
        return fba, det

def presence_tau(n_frag, n_bits, alpha=0.01):
    """Zero-bit threshold when a k-fragment configuration runs k+1 zero-bit tests (one per fragment,
    plus the fused codeword) inside one false-positive budget: each test gets alpha/(k+1). A single
    fragment's fused codeword IS its own, so it runs one test at alpha."""
    n_tests = 1 if n_frag <= 1 else n_frag + 1
    return float(binom.ppf(1.0 - alpha / n_tests, n_bits, 0.5) + 1) / n_bits

def presence_detected(per_ba, fused_ba, n_frag, n_bits=100, alpha=0.01):
    """Best-path presence: any fragment, or the fusion, above the budget-split threshold."""
    tau = presence_tau(n_frag, n_bits, alpha)
    return bool(fused_ba >= tau or any(v >= tau for v in per_ba.values()))

def _id_payload(image_id, nb):
    from src.payload import image_id_to_payload
    return image_id_to_payload(image_id, n_bits=nb)

# ---------------- baseline wrapper ----------------
class BaselineMethod:
    def __init__(self, m):
        self.m = m; self.name = m.name; self.n_bits = int(m.n_bits)
        self.tau = float(binom.ppf(0.99, self.n_bits, 0.5) + 1) / self.n_bits

    def secret_for(self, idx):
        return np.random.RandomState(idx).randint(0, 2, self.n_bits).astype(np.uint8)

    def embed(self, pil, idx):
        bits = self.secret_for(idx)
        return self.m.embed(pil, bits), bits

    def decode(self, att, bits):
        rec = np.asarray(self.m.decode(att)).astype(np.uint8)
        n = min(len(rec), len(bits)); ba = float(np.mean(rec[:n] == bits[:n])) if n else 0.0
        return ba, (ba >= self.tau)

class SyncWrapped:
    """SyncSeal as a shared, method-agnostic geometric front-end.

    SyncSeal only adds a sync watermark and perspective-unwarps the frame; it carries no payload and
    is independent of which watermark is underneath. It is therefore available to EVERY method, not
    just ours, and withholding it from the baselines (or from us) would misattribute geometry
    robustness. Decode is `rectify-always, decode-once`: exactly one decode attempt per image, at the
    method's unchanged threshold, so the front-end cannot buy detection with false positives. (Our
    composite's own cascade instead SEARCHES, which is sound only because a keyed verification gates
    acceptance; that difference is reported separately, not hidden inside this wrapper.)"""
    def __init__(self, inner, dev):
        self.inner = inner; self.name = inner.name + "+sync"
        self.n_bits = inner.n_bits; self.tau = getattr(inner, "tau", None); self.dev = dev
        from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
        self._sync = load_sync(dev=dev); self._embed_sync = sync_embed; self._rectify = sync_rectify

    def secret_for(self, idx):
        return self.inner.secret_for(idx)

    def embed(self, pil, idx):
        img, sec = self.inner.embed(pil, idx)
        return self._embed_sync(self._sync, img, self.dev), sec

    def decode(self, att, sec):
        try:
            rect, _ = self._rectify(self._sync, att, self.dev)
        except Exception:
            rect = att                                  # rectification failed -> decode as received
        return self.inner.decode(rect, sec)


def measure_fpr(method, imgs, tmp, n_neg):
    """Empirical false-accept rate on UNWATERMARKED covers, through the method's own decode path
    (front-end included). Makes any FPR the front-end costs visible instead of assumed."""
    acc = 0; tot = 0
    for i, fp in enumerate(imgs[:n_neg]):
        cover = Image.open(fp).convert("RGB").resize((512, 512))
        try:
            _, det = method.decode(cover, method.secret_for(10_000 + i))
            acc += int(bool(det)); tot += 1
        except Exception:
            pass
    return (acc / tot if tot else None), tot


# ---------------- per-method run ----------------
def run_method(method, imgs, attacks, attack_one, tmp, out_json):
    t0 = time.time()
    embeds, secrets, psnrs, ssims = [], [], [], []
    for i, fp in enumerate(imgs):
        cover = Image.open(fp).convert("RGB").resize((512, 512))
        emb, sec = method.embed(cover, i)
        if emb.size != (512, 512): emb = emb.resize((512, 512))
        embeds.append(emb); secrets.append(sec)
        ca = np.asarray(cover, np.float32) / 255.0; ea = np.asarray(emb, np.float32) / 255.0
        psnrs.append(psnr(ca, ea)); ssims.append(ssim(ca, ea))
    R = {a: {"ba": [], "det": []} for a in attacks}
    frame_det = {a: [] for a in attacks if a == "img2video"}   # per-frame detection fraction
    for i, emb in enumerate(embeds):
        ip = os.path.join(tmp, f"e_{i}.png"); emb.save(ip)
        for a in attacks:
            try:
                if is_adv(a):                                 # in-env, PIL-based, list of frames
                    frames = apply_adv(a, emb, i)
                    bas, dets = [], []
                    for fr in frames:
                        if fr.size != (512, 512): fr = fr.resize((512, 512))
                        b, d = method.decode(fr, secrets[i]); bas.append(b); dets.append(float(d))
                    R[a]["ba"].append(float(np.mean(bas)))
                    R[a]["det"].append(float(any(dets)))       # video: detected if ANY frame carries it
                    if a == "img2video": frame_det[a].append(float(np.mean(dets)))
                else:                                          # standard path-based attack
                    op = os.path.join(tmp, f"e_{i}_{a}.png")
                    attack_one(a, ip, op)
                    att = Image.open(op).convert("RGB")
                    if att.size != (512, 512): att = att.resize((512, 512))
                    ba, det = method.decode(att, secrets[i])
                    R[a]["ba"].append(ba); R[a]["det"].append(float(det))
            except Exception as e:
                print(f"  [{method.name}][{a}] img{i} FAIL {type(e).__name__}:{str(e)[:55]}", flush=True)
            finally:
                for f in glob.glob(os.path.join(tmp, f"e_{i}_{a}*")):
                    try: os.remove(f)
                    except Exception: pass
        try: os.remove(ip)
        except Exception: pass
        if (i + 1) % 10 == 0: print(f"  {method.name} {i+1}/{len(embeds)} ({time.time()-t0:.0f}s)", flush=True)
    mean = lambda x: float(np.mean(x)) if x else None
    out = {"method": method.name, "n_bits": method.n_bits, "n": len(embeds),
           "tau": getattr(method, "tau", None), "psnr": mean(psnrs), "ssim": mean(ssims),
           "sec": time.time() - t0,
           "attacks": {a: ({"bit_acc": mean(R[a]["ba"]), "tpr": mean(R[a]["det"])}
                            | ({"frame_det": mean(frame_det[a])} if a == "img2video" else {}))
                       for a in attacks}}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"[{method.name}] psnr={out['psnr']:.2f} ssim={out['ssim']:.3f}  {out['sec']:.0f}s -> {out_json}", flush=True)
    return out

def build_method(name, dev, tm_variant, geo=False, config=None):
    if name == "ours":
        return OursComposite(dev, tm_variant, geo=geo, vine_variant="R", config=config)
    if name == "ours_b":                                   # Ours with the higher-PSNR VINE-B fragment
        return OursComposite(dev, tm_variant, geo=geo, vine_variant="B", config=config)
    from wbench.methods import build_methods
    reg = build_methods([name], dev)
    if name not in reg: raise RuntimeError(f"method {name} failed to load")
    return BaselineMethod(reg[name])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--image_glob", default="*.png")
    p.add_argument("--n_images", type=int, default=200)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--methods", nargs="+",
                   default=["ours", "dwtDct", "dwtDctSvd", "rivaGan", "trustmark_b", "vine_b", "vine_r"])
    p.add_argument("--attacks", nargs="+", default=ALL_ATTACKS)
    p.add_argument("--tm_variant", default="B")
    # --geo turned three independent decisions into one switch, which is what the solver is for.
    # It stays only as a shorthand for "every stage on", and --frontend names them individually.
    p.add_argument("--geo", action="store_true",
                   help="shorthand for --frontend resync scale angle (every stage on)")
    p.add_argument("--frontend", nargs="*", default=None, choices=["resync", "scale", "angle"],
                   help="geometric front-end stages to enable, chosen individually; the method "
                        "returns these per request rather than bundling them")
    p.add_argument("--config", default=None,
                   help="JSON file OR inline JSON with the solver's configuration "
                        "{frags, order, strengths, resync, nested}; when given, the Ours row is the "
                        "configuration the solver returned for a request rather than a pinned point")
    p.add_argument("--syncseal", action="store_true",
                   help="wrap EVERY method in the shared SyncSeal front-end (rectify-always, decode-once)")
    p.add_argument("--fpr_negatives", type=int, default=0,
                   help="also measure empirical false-accept rate on this many unwatermarked covers")
    p.add_argument("--out_dir", default=f"{WORKSPACE}/eval_dev200")
    args = p.parse_args()
    dev = "cuda"
    free, tot = torch.cuda.mem_get_info(0); print(f"[gpu] free {free/1e9:.1f}/{tot/1e9:.1f} GB", flush=True)
    imgs = sorted(glob.glob(os.path.join(args.image_dir, args.image_glob)))[args.start_idx:args.start_idx + args.n_images]
    print(f"[eval] {len(imgs)} imgs; methods={args.methods}; {len(args.attacks)} attacks", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    attack_one = build_attack_one(dev)
    cfg = None
    if args.config:
        cfg = json.load(open(args.config)) if os.path.exists(args.config) else json.loads(args.config)
        print(f"[config] solver configuration: {cfg}", flush=True)
    for mname in args.methods:
        oj = os.path.join(args.out_dir, f"{mname}.json")
        if os.path.exists(oj):
            print(f"[skip] {mname} (exists)", flush=True); continue
        try:
            if args.frontend is not None:
                fe = set(args.frontend)
                cfg = dict(cfg or {})
                cfg.setdefault("resync", "resync" in fe)
                cfg.setdefault("nested", "scale" in fe)
                cfg.setdefault("scale_search", "scale" in fe)
                cfg.setdefault("angle_sweep", "angle" in fe)
            method = build_method(mname, dev, args.tm_variant, geo=args.geo, config=cfg)
            if args.syncseal:
                method = SyncWrapped(method, dev)
            res = run_method(method, imgs, args.attacks, attack_one, tmp, oj)
            if args.fpr_negatives:
                fpr, ntot = measure_fpr(method, imgs, tmp, args.fpr_negatives)
                res["fpr_empirical"] = fpr; res["fpr_n"] = ntot
                json.dump(res, open(oj, "w"), indent=2)
                print(f"[{method.name}] empirical FPR {fpr:.4f} on {ntot} unwatermarked covers", flush=True)
            del method; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{mname}] METHOD FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print("EVAL_MATRIX_DONE", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
