#!/bin/bash
# Sourced by every job script under experiments/. experiments/README.md says what each location holds.
: "${TAILOR_PROJECT:?set TAILOR_PROJECT to the repository root}"
: "${TAILOR_OUTPUTS:?set TAILOR_OUTPUTS to the folder that holds the campaign run tree}"
export WM_RUN_ROOT="${WM_RUN_ROOT:-$TAILOR_OUTPUTS/rigor_editing_only_20260911/unified_ba_20260914}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
