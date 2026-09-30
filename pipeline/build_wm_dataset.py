#!/usr/bin/env python
"""Build the ~10k multi-distribution watermark evaluation pool (splits A-E).

A MS-COCO (real photos) | B DiffusionDB SD-1.5 (+prompt) | C UltraEdit (local, real+edit)
D DALL-E3 1024 (+caption) | E DIV2K + Unsplash-lite (high-res real).

Design in exp_plan.tex Sec. Datasets. Post-hoc methods watermark these directly;
in-gen baselines re-generate from the B/D prompts. Native resolution preserved
(resize to 256/512/1024 happens at eval time). Deterministic: --n gives a prefix,
so a smoke subset is a strict prefix of the full build.
"""
import os, sys, csv, json, argparse, io
from PIL import Image

SC = "/data/tailor/workspace"
LOCAL_ULTRAEDIT = f"{SC}/ultraedit_10k/source"
MAXSIDE = 2048  # cap high-res (E) long side to bound disk

# key, hf-name (or __local__), config, split, distribution, full target N
SOURCES = [
    ("A", "detection-datasets/coco",                                          None, "val",   "real_photo",            2500),
    ("B", "svjack/diffusiondb_random_10k",                                    None, "train", "sd15_generated",        2500),
    ("C", "__local_ultraedit__",                                             None, None,    "real_edit",             2500),
    ("D", "ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions",    None, "train", "dalle3_generated",      1500),
    ("E", "bdanko/DIV2K_train_HR",                                            None, "train", "highres_real_div2k",     500),
    ("E", "danjacobellis/LSDIR",                                             None, "train", "highres_real_lsdir",     500),
]

def first_image(ex):
    for k, v in ex.items():
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, dict) and "bytes" in v and v["bytes"]:
            try: return Image.open(io.BytesIO(v["bytes"]))
            except Exception: pass
    return None

def first_prompt(ex):
    for k, v in ex.items():
        if isinstance(v, str) and any(t in k.lower() for t in ("prompt", "caption", "text")):
            return v
    j = ex.get("json")
    if j is not None:
        try:
            d = j if isinstance(j, dict) else json.loads(j)
            for kk in ("prompt", "long_caption", "caption", "description", "txt"):
                if kk in d and d[kk]:
                    return str(d[kk])
        except Exception:
            pass
    return ""

def save_img(img, path):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAXSIDE:
        s = MAXSIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img.save(path, "PNG")
    return w, h  # native (pre-cap) size

def local_ultraedit(cap):
    import glob
    files = sorted(glob.glob(f"{LOCAL_ULTRAEDIT}/*.png"))[:cap]
    for fp in files:
        yield {"image": Image.open(fp), "__src_path__": fp}

def build(out, ncap, seed):
    os.makedirs(out, exist_ok=True)
    man_path = f"{out}/manifest.csv"
    mf = open(man_path, "w", newline="")
    w = csv.writer(mf); w.writerow(["id", "split", "source", "distribution", "native_w", "native_h", "path", "prompt"])
    counts, meta = {}, {}
    for key, name, cfg, split, dist, tgt in SOURCES:
        n = min(tgt, ncap) if ncap else tgt
        d = f"{out}/{key}"; os.makedirs(f"{d}/img", exist_ok=True)
        print(f"\n=== [{key}] {name}  target={n}  dist={dist} ===", flush=True)
        # get iterator
        if name == "__local_ultraedit__":
            it = local_ultraedit(n)
        else:
            from datasets import load_dataset
            ds = load_dataset(name, cfg, split=split, streaming=True) if cfg else load_dataset(name, split=split, streaming=True)
            it = iter(ds)
        got = 0; tried = 0
        base = counts.get(key, 0)
        while got < n:
            try:
                ex = next(it)
            except StopIteration:
                print(f"  ! stream exhausted at {got}/{n}", flush=True); break
            tried += 1
            if tried > n * 20 + 100:
                print(f"  ! give up after {tried} tries, got {got}", flush=True); break
            img = first_image(ex)
            if img is None:
                continue
            idx = base + got
            iid = f"{key}_{idx:05d}"
            path = f"{d}/img/{iid}.png"
            try:
                nw, nh = save_img(img, path)
            except Exception as e:
                print(f"  skip decode err: {type(e).__name__}", flush=True); continue
            prompt = first_prompt(ex).replace("\n", " ").strip()[:500]
            w.writerow([iid, key, name, dist, nw, nh, path, prompt])
            got += 1
            if got % max(1, n // 5) == 0:
                print(f"  {got}/{n}  last={nw}x{nh}  prompt={'Y' if prompt else '-'}", flush=True)
        counts[key] = base + got
        meta[f"{key}:{name}"] = {"got": got, "dist": dist, "split": split}
        mf.flush()
    mf.close()
    json.dump({"seed": seed, "ncap": ncap, "counts": counts, "sources": meta},
              open(f"{out}/meta.json", "w"), indent=2)
    print("\n===== SUMMARY =====")
    for k in ["A", "B", "C", "D", "E"]:
        print(f"  {k}: {counts.get(k,0)}")
    print(f"  TOTAL: {sum(counts.values())}")
    print(f"  manifest: {man_path}")
    print("BUILD_DONE")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{SC}/wm_dataset10k/pool")
    ap.add_argument("--n", type=int, default=0, help="per-source cap (0 = full targets)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.n, a.seed)
    os._exit(0)  # skip torch/pyarrow atexit finalizer (harmless GIL abort noise)
