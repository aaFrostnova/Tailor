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

Embedding and live calibration additionally need PyTorch, the three watermark
fragments and the attack models:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/solver:$PWD/measurement:$PWD/pipeline
```

The fragments (VINE, TrustMark, VideoSeal) and the attack models (diffusion
regeneration, neural compression, image editing, watermark removal) come from
their upstream projects; `requirements.txt` names them.

## Usage

### Solve a request

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 --min_psnr 42 --max_ms 100
```

```
QUERY custom: PSNR>=42.0 ms<=100.0 ba>=0.9 bits>=0 ['jpeg25']
  VINE(alpha=0.3)   PSNR~45.6dB . 57ms
```

A compression-only request is answered by one fragment at low strength. Harden
the request and the composition grows:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 crop75 regen --min_psnr 38 --max_ms 500
```

```
  VINE(alpha=0.7) + TrustMark   PSNR~38.3dB . 70ms
```

Ask for more than the library can deliver and the answer is a statement about
the library, not a failed search:

```bash
python solver/watermark_smt_v2.py --attacks jpeg25 crop75 crop50 rot9 regen rinse --min_psnr 46 --max_ms 200
```

```
  UNSAT - no combination satisfies these conditions
```

### The request

| Flag | Meaning |
|---|---|
| `--attacks` | the attacks the configuration must survive |
| `--min_psnr` | the fidelity floor in dB |
| `--max_ms` | the latency ceiling in ms, embedding plus verification |
| `--min_ba` | the bit accuracy a verification must reach, which the FPR budget sets |

Available attacks: `jpeg25`, `blur`, `noise`, `bright`, `contrast`, `rs256`,
`hflip`, `crop75`, `crop50`, `rot9`, `crop_jpeg`, `border20`, `vaeB`, `vaeC`,
`regen`, `rinse2x`, `ctrlregen_s03`, `ctrlregen_s05`, `editing_ip2p_s20_v1`,
`unmarker`.

### The answer

The solver returns the fragments, the strength of each, their embedding order,
and the geometric stage if one is enabled, together with the PSNR and latency it
predicts. A configuration with more verification paths must clear a stricter
threshold, because they share the request's false-positive budget, so adding a
fragment is not free even when it adds coverage.

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
```

`inputs/surrogate_canonical.json` is the database: 60 recovery curves over
fragment strength, 120 interference curves per ordered pair and attack, 827
curves for the geometric stages, the per-image scores behind every mean, and the
distortion, latency and capacity entries. `inputs/requests.json` holds the 7,321
requests the paper evaluates, each one the four inputs above.

The image pool and the model checkpoints are not redistributed here. The first
is rebuilt by `pipeline/build_wm_dataset.py`; the second come from the upstream
projects.

## License

MIT, see `LICENSE`. The watermark fragments and the attack models are used
through their upstream packages and are not redistributed here; their own
licenses govern those packages.
