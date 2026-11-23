# Experiments

This folder holds code of the experiments reported in the paper, as it ran, for
the 7,321-request benchmark. No result is included: no measurement, cell cache, log,
image or table. For the release, locations on the machine the runs used now come
from the environment variables below; paths into the original development tree
now point at this repository's layout (`solver/`, `inputs/`, `wbench/`,
`experiments/attack/`); some folder, file, package and job names were renamed;
cluster partition names are generic (`cpu`, `gpu`); and run bookkeeping that the
experiments do not need was left out. The code still refers to its runs by their
dated folder names.

## Locations

| Variable | Points at |
| --- | --- |
| `TAILOR_PROJECT` | this repository |
| `TAILOR_WORKSPACE` | the image pool and the measurement workspace, as in the top-level README |
| `TAILOR_OUTPUTS` | the folder holding the campaign run tree `rigor_editing_only_20260911/unified_ba_20260914` |
| `WM_RUN_ROOT` | that run tree; `runtime.sh` derives it from `TAILOR_OUTPUTS` |
| `TAILOR_RUNS` | the folder holding the follow-up runs |
| `TAILOR_MAIN_CONTROLS_RUN` | one follow-up run; its `root/` is a byte-identical copy of the campaign runner |
| `TAILOR_PYTHON` | the interpreter of the main environment (default `python`) |
| `TAILOR_CTRLREGEN_PYTHON`, `TAILOR_UNMARKER_PYTHON` | the interpreters of the two attack environments |
| `TAILOR_SD21`, `TAILOR_VINE_REPO`, `HF_HOME` | as in the top-level README |
| `TAILOR_PAPER`, `TAILOR_PAPER_COMMIT` | the repository holding the cohort files (`data/main_cohort.json`, `data/ablation_evaluation.json`, `data/request_specifications.json`, `data/editing_requests.json`) and the tables that some steps read, and the commit they were pinned at |

Every job script sources `runtime.sh`, which checks the two required locations
and pins thread counts. The jobs are written for SLURM.

## How the campaign code runs

Each folder under `campaign/` ran from its own place in the campaign run tree:
`campaign/code` as `$WM_RUN_ROOT/code`, `campaign/fixed/code` as
`$WM_RUN_ROOT/fixed/code`, and so on. Copy a folder there before running it. The
runner imports the fragments, the attacks and the verifier from this repository
(`src/`, `solver/`, `wbench/`), so `TAILOR_PROJECT` must point here.

The follow-up runs never edit the campaign. `main_controls/relocate.py` copies the
runner into a new run's `root/`, and every follow-up imports it from there.

The campaign's own inputs (the request sets and the frozen tables under
`fixed/inputs/`) belong to its run tree and are not included. `inputs/` at the
top level holds the same tables with machine paths and commit ids blanked, so
their digests differ from the ones some steps pin.

## What produced what

Reported in the paper:

| Paper | Code |
| --- | --- |
| Main comparison, TAILOR-F | `campaign/code`, `campaign/fixed/code`, `campaign/fallback/code` (margin fallback), `campaign/operations` (supervision and evaluation export) |
| Main comparison, TAILOR-G | `campaign/shared_table_20260916` (variant `G_shared`) |
| Main comparison, VINE, TrustMark, VideoSeal | `campaign/single_baselines` |
| Main comparison, Optimized Single, Always All Three | `main_controls` (also their SSIM, `main_controls/measure_ssim.py`) |
| Ablation studies, recovery-interaction sensitivity | `campaign/shared_table_20260916/priority_20260917` (the variants of one shared model, listed in its `code/common.py`), `campaign/component_ablation_20260923` (single-fragment and unit-strength restrictions) |
| Screening margin | the margin arms of `campaign/fixed/code` and `campaign/fallback/code` |
| Grid Enumeration and the fidelity-threshold figure | `grid`, with the knot distortions from `grid_measure/measure_distortion.py` |
| Direct fragment superposition | `appendix/fragment_superposition` |
| Fresh deployment identities | `heldout` |
| Where each fragment writes its payload | `appendix/request_examples/ring_probe` |
| Qualitative request examples and their timing | `appendix/request_examples`, `appendix/qualitative_timing` |

Run during the work and not reported in the paper: `campaign/baselines`
(one-proposal rule-based baselines), `fpr` (false acceptances of deployed
configurations on fresh negative images), `grid_measure` (ordered pairwise
interference and a distortion diagnostic), `scaling` (solver time against the
number of fragments, with virtual fragments) and `smt_fast` (an exact-output
acceleration of the top-k search).

Shared by all of the above: `tailor_exp` (request cohorts, result schema, run
registry, the table gate), `tools/export_staging.py` (tables and figures from the
analysis files) and `tests` (unit tests; most read the run records and cohort
files, and fail or skip without them).

`attack/` holds the CtrlRegen+ and UnMarker drivers these runs called.
`scripts/attack/` at the top level holds standalone rewrites of the same two
attacks for use outside the campaign.
