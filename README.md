# Made to Measure: Designing Image Watermarks to Specification

TAILOR takes a deployment request and returns a watermark configuration that
meets it, or says the library cannot. A request is four inputs: the attacks the
watermark must survive, the false-positive budget a single verification may
spend, the PSNR floor, and the latency ceiling. TAILOR searches fragment
subsets, their embedding order, a continuous strength per fragment, and an
optional geometric recovery stage, subject to all four at once, and returns the
lowest-distortion configuration that satisfies them.

![Overview of TAILOR](assets/overview.png)

Offline characterization builds a performance database of attack coverage,
distortion and runtime. Joint configuration selection solves an SMT model of the
request over that database. Live calibration validates the chosen configuration
on your own images under the requested attacks before it is deployed.

## Install

The solver needs three packages:

```bash
git clone <this repository>
cd Tailor
pip install z3-solver numpy scipy
```

That is enough to solve requests: the measured database the solver reads is in
`inputs/`.

Embedding and live calibration additionally need PyTorch and the models:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/solver:$PWD/measurement:$PWD/pipeline
python tools/fetch_models.py
```

### Models

No model is redistributed here. `tools/fetch_models.py` puts each one where
`src/paths.py` looks for it, under `external/` by default, and `--check` reports
what is present without fetching anything.

| Component | What it is | How it arrives |
| --- | --- | --- |
| TrustMark | fragment | `pip install trustmark`, which fetches its own weights |
| VINE | fragment | cloned from its project; the encoder and decoder weights are two Hugging Face repositories the classes fetch themselves |
| VideoSeal | fragment | cloned from its project, which resolves its own model card |
| WatermarkAttacker | the regeneration attack | cloned from its project; carries no weights |
| Stable Diffusion 2.1, 1.5, 1.4 | the regeneration and VAE attacks | `--only sd`, resolved by identifier and cached on first use. The 2.1 repositories are gated: accept the terms on the model page and `hf auth login`, or point `TAILOR_SD21` at a local directory |
| SyncSeal | the learned geometric stage | placed by hand. `--only syncseal` prints where `syncmodel.jit.pt` goes; only the stage that rectifies before decoding uses it |
| MaskWM | a baseline | placed by hand, likewise. Nothing in the method needs it |

`TAILOR_MODELS` moves the whole tree; a per-component variable moves one of
them, and `src/paths.py` lists all of them in one place. The same file decides
where a measurement campaign writes (`TAILOR_WORKSPACE`) and where the image
pool is (`TAILOR_POOL`), so nothing outside `inputs/` is a fixed path.

CtrlRegen and UnMarker need dependencies that conflict with this environment, so
they are invoked out of process: install each in its own environment and name
its interpreter through `TAILOR_CTRLREGEN_PYTHON` and `TAILOR_UNMARKER_PYTHON`.

## Usage

### Solve a request

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 --fpr 1e-2 --min_psnr 42 --max_ms 100
```

```
QUERY custom: PSNR>=42.0 ms<=100.0 FPR<=0.01 (ba>=0.63) bits>=0 ['jpeg25']
  VINE(alpha=0.3)   PSNR~45.6dB . 57ms
```

A compression-only request is answered by one fragment at low strength. Harden
the attack set and the composition grows:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 crop75 regen --fpr 1e-6 --min_psnr 38 --max_ms 500
```

```
QUERY custom: PSNR>=38.0 ms<=500.0 FPR<=1e-06 (ba>=0.74) bits>=0 ['jpeg25', 'crop75', 'regen']
  VINE(alpha=0.7) + TrustMark   PSNR~38.3dB . 70ms
```

Ask for more than the library can deliver and the answer is a statement about
the library, not a failed search:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 crop75 crop50 rot9 regen rinse --fpr 1e-6 --min_psnr 46 --max_ms 200
```

```
  UNSAT - no combination satisfies these conditions
```

### The request is four inputs

| Flag | The input |
|---|---|
| `--attacks` | the attacks the configuration must survive |
| `--fpr` | the false-positive budget one verification of one image may spend |
| `--min_psnr` | the fidelity floor in dB |
| `--max_ms` | the latency ceiling in ms, embedding plus verification |

Available attacks: `jpeg25`, `blur`, `noise`, `bright`, `contrast`, `rs256`,
`hflip`, `crop75`, `crop50`, `rot9`, `crop_jpeg`, `border20`, `vaeB`, `vaeC`,
`regen`, `rinse2x`, `ctrlregen_s03`, `ctrlregen_s05`, `editing_ip2p_s20_v1`,
`unmarker`.

The budget is spent, not assumed. `--fpr` sets the bit accuracy a verification
must reach, printed beside it: a budget of $10^{-2}$ asks for 0.63, $10^{-6}$
for 0.74, $10^{-9}$ for 0.80. It is a real constraint, and tightening it alone
can decide a request:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 regen --fpr 1e-1 --min_psnr 40 --max_ms 300
#   TrustMark   PSNR~41.3dB . 12ms
python solver/watermark_smt_v2.py --attacks jpeg25 regen --fpr 1e-2 --min_psnr 40 --max_ms 300
#   UNSAT
```

A configuration that runs more verification paths must clear a stricter
threshold, because the paths share this one budget, so a fragment is never free
even when it adds coverage. `--min_ba` states the required bit accuracy directly
instead of deriving it from a budget; give one or the other.

### The answer

The solver returns the fragments, the strength of each, their embedding order,
and the geometric stage if one is enabled, together with the PSNR and latency it
predicts.

### Validate on your own images

The database describes the images it was measured on, which may not behave like
yours. Live calibration runs the full embed, attack and decode path on your
images and deploys only what passes there:

```bash
python solver/live_topk_full.py enumerate <class> <n_requests>   # candidates per request
python solver/live_topk_full.py measure   <class> [N=100]        # GPU: embed, attack, decode
python solver/live_topk_full.py walk      <class> [N=100]        # verdicts, no GPU
python solver/live_topk_full.py patch     <class> <shard> <n>    # refine and re-solve on failure
```

`measure` writes one record per configuration and attack under
`$TAILOR_WORKSPACE/live_topk/cells/` and reuses anything already there, so a
campaign is resumable and a configuration measured once is free for every later
request that selects it.

### Rebuild the database for your own fragments

To characterize a different fragment library, run the offline stages and refit:

```bash
python pipeline/build_wm_dataset.py --out $TAILOR_WORKSPACE/pool   # the image pool
python pipeline/perimage_campaign.py                               # recovery, per image
python pipeline/make_delta_curves.py                               # interference, per ordered pair
python pipeline/make_frontend_components.py                        # the geometric stages
python pipeline/fit_surrogate_curves.py                            # knots to curves
python pipeline/make_canonical_surrogate.py                        # -> inputs/surrogate_canonical.json
```

## What is in the box

```
solver/        the SMT model, the request protocol, and the live-calibration loop
measurement/   the verification rule, the FPR accounting, and the measurement harness
src/           the watermark fragments, the geometric stages, the attacks, soft decoding
pipeline/      the campaigns that build the performance database
inputs/        the measured database the solver reads
tools/         fetching the models, which are not redistributed
```

`inputs/surrogate_canonical.json` is the database: 60 recovery curves over
fragment strength, 120 interference curves per ordered pair and attack, 827
curves for the geometric stages, the per-image scores behind every mean, and the
distortion, latency and capacity entries. `inputs/requests.json` holds the 7,321
requests the paper evaluates, each one the four inputs above.

The image pool and the models are not redistributed here. The pool is rebuilt by
`pipeline/build_wm_dataset.py` and the models arrive through
`tools/fetch_models.py`, as **Models** above describes.

## License

MIT, see `LICENSE`. The watermark fragments and the attack models are used
through their upstream packages and are not redistributed here; their own
licenses govern those packages.
