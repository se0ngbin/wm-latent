#!/usr/bin/env bash
# Consistent OOD margin matrix: self-contained JEPA-static + Dreamer, all axes,
# seeds. One renderer per stack (train==eval), fresh heads, rk=0. Aggregate with
# ood_seed_aggregate.py. Usage: ood_matrix.sh ["axes"] ["seeds"]
set -uo pipefail
AXES=${1:-"color shape rotate"}
SEEDS=${2:-"0 1 2"}
LEWM=/home/seongbin/latent/le-wm
LCBF=/home/seongbin/latent/latent_cbf
DR=/data/seongbin/dreamer/dreamer
LOG=/home/seongbin/latent/run_logs
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
declare -A WM=( [baseline]="$DR/enc_lip_sweep/baseline/rssm_ckpt.pt"
                [enc-lip]="$DR/enc_lip_sweep_w1/image/rssm_ckpt.pt"
                [invariance]="$DR/enc_lip_sweep_w1/invariance/rssm_ckpt.pt" )

for axis in $AXES; do
  for s in $SEEDS; do
    echo "[$(date +%H:%M:%S)] JEPA-static $axis seed=$s"
    CUDA_VISIBLE_DEVICES=0 "$LEWM/.venv/bin/python" -u "$LEWM/scripts/ood_margin_jepa_static.py" \
      "$axis" "$s" > "$LOG/ood_margin_jepa_static_${axis}_s${s}.log" 2>&1
    grep -E "KEY jepa" "$LOG/ood_margin_jepa_static_${axis}_s${s}.log"
    for tag in baseline enc-lip invariance; do
      echo "[$(date +%H:%M:%S)] Dreamer $tag $axis seed=$s"
      CUDA_VISIBLE_DEVICES=0 "$LCBF/.venv/bin/python" -u \
        "$LCBF/src/latent_cbf/scripts/ood_margin_dreamer.py" \
        --rssm_ckpt "${WM[$tag]}" --axis "$axis" --tag "$tag" --seed "$s" \
        > "$LOG/ood_margin_dreamer_${axis}_${tag}_s${s}.log" 2>&1
      grep -E "KEY dreamer" "$LOG/ood_margin_dreamer_${axis}_${tag}_s${s}.log"
    done
  done
done
echo "OOD_MATRIX_DONE"
