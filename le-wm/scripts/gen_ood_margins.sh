#!/usr/bin/env bash
# Generate per-appearance margins (margin_gp+nogp) for the RETRAINED value-function
# column: 3 JEPA models x {purple, diamond, rot90}, in margin_heads.pt format.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
GPU=${1:-1}
MDIR=/data/seongbin/dreamer/lewm/_value_ood/margins; LOG=/home/seongbin/latent/run_logs
mkdir -p "$MDIR" "$LOG"
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
for tag in baseline jacobian jacpull; do
  w=${WM[$tag]}
  for cond in purple diamond rot90; do
    out="$MDIR/${tag}_${cond}.pt"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "[$(date +%H:%M:%S)] margin $tag/$cond"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/save_ood_margin_lewm.py \
      --lewm_ckpt /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
      --cond "$cond" --run_name "$w" --out "$out" \
      > "$LOG/genmargin_${tag}_${cond}.log" 2>&1
    grep "KEY oodmargin" "$LOG/genmargin_${tag}_${cond}.log" || tail -3 "$LOG/genmargin_${tag}_${cond}.log"
  done
done
echo GEN_MARGINS_DONE
