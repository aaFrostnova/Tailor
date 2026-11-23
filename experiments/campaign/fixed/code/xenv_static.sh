#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=cf_xenv
#SBATCH -t 0-12:00
#SBATCH -o logs/certify_full_xenv_%A_%a.out
# Cross-environment chain for the groups of one class that name CtrlRegen+ / UnMarker: $1 class index,
# $2 CtrlRegen+ images, $3 UnMarker images; the array index enumerates the groups with xenv_needed.
set -euo pipefail
CF="${TAILOR_PROJECT:?}"
SC="${TAILOR_WORKSPACE:?}"
CRPY="${TAILOR_CTRLREGEN_PYTHON:?}"
UMPY="${TAILOR_UNMARKER_PYTHON:?}"
FPPY="${TAILOR_PYTHON:-python}"
KEYS=(C1 C2 C3 C4 C5); K=${KEYS[$1]}; NC=${2:-100}; NU=${3:-30}
# CERT_DOMAIN (exported by the caller) selects the domain's group directory; certify_full.py decode_xenv reads it too
DOM=${CERT_DOMAIN:-indomain}; CDIR="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/certify_full; [ "$DOM" != "indomain" ] && CDIR="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/certify_full_$DOM
cd $CF
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub
export PYTHONPATH="$CF:$CF/scripts:$CF/solver:$CF/external/WatermarkAttacker:$SC"
# the i-th group that needs the chain, and what it needs
GI=$FULL_GIDX; STEPS=$FULL_STEPS; UNM=$FULL_UNM
GD=$CDIR/$K/g$(printf %04d $GI)
echo "node=$(hostname) class=$K group=$GI steps=$STEPS unmarker=$UNM NC=$NC NU=$NU $(date)"
[ "$(ls $GD/embed/*.png 2>/dev/null | wc -l)" -ge 1 ] || { echo "XENV_ABORT: no embeds in $GD"; exit 1; }
if [ "$STEPS" != "-" ]; then
  mkdir -p $GD/embed_cr; ls $GD/embed/*.png | head -n $NC | xargs -I{} ln -sf {} $GD/embed_cr/
  $CRPY experiments/attack/ctrlregen_batch.py --in_dir $GD/embed_cr --out_dir $GD/att_ctrlregen --step_list $STEPS 2>&1 | tail -20
fi
if [ "$UNM" = "1" ] && [ "$(ls $GD/att_unmarker/*.png 2>/dev/null | wc -l)" -lt "$NU" ]; then
  $UMPY experiments/attack/unmarker_batch.py --in_dir $GD/embed --out_dir $GD/att_unmarker --config attack_configs/Vine.yaml --n $NU --start_idx 0 --batch 1 2>&1 | tail -20
fi
"$FPPY" -u "${TAILOR_WORKSPACE:?}"/attack_extension_nfpa_edit_20260911/code/xenv_attacks.py "$CDIR/$K/g$(printf %04d $GI).json"
$FPPY -u "${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/fixed/code/certify_full_frozen.py decode_xenv "$1" "$GI" 2>&1
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
