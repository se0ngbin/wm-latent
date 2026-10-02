#!/usr/bin/env bash
# Dreamer/RSSM obstacle-appearance margin-OOD test (analog of ood_margin.sh).
# Three RSSM variants: baseline (no reg) / enc-lip (encoder-Lipschitz, ~jacobian) /
# invariance (~pull). Trains M_og + M_new per variant, evals both on both. Axis
# arg = color|shape. Usage: ood_margin_dreamer.sh <color|shape>
set -uo pipefail
AXIS=${1:-color}
SEED=${2:-0}
LCBF=/home/seongbin/latent/latent_cbf
DR=/data/seongbin/dreamer/dreamer
LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm

declare -A WM=(
  [baseline]="$DR/enc_lip_sweep/baseline/rssm_ckpt.pt"
  [enc-lip]="$DR/enc_lip_sweep_w1/image/rssm_ckpt.pt"
  [invariance]="$DR/enc_lip_sweep_w1/invariance/rssm_ckpt.pt"
)

for tag in baseline enc-lip invariance; do
  echo "[$(date +%H:%M:%S)] dreamer $tag axis=$AXIS seed=$SEED"
  CUDA_VISIBLE_DEVICES=0 "$LCBF/.venv/bin/python" -u \
    "$LCBF/src/latent_cbf/scripts/ood_margin_dreamer.py" \
    --rssm_ckpt "${WM[$tag]}" --axis "$AXIS" --tag "$tag" --seed "$SEED" \
    > "$LOG/ood_margin_dreamer_${AXIS}_${tag}_s${SEED}.log" 2>&1
  grep -vE "Warning|warn|Gym|migration|deprecat|INFO|Encoder|Decoder|Optimizer|shapes|_scaler" \
    "$LOG/ood_margin_dreamer_${AXIS}_${tag}_s${SEED}.log" | grep -E "===|M_|KEY|encoded"
done
echo "OOD_MARGIN_DREAMER_${AXIS}_s${SEED}_DONE"
