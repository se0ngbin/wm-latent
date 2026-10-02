#!/usr/bin/env bash
# OOD matrix on the REAL instantaneous margin (margin_gp + margin_nogp), original
# latent_cbf loss/config. Models: JEPA {baseline,jacobian,jac+pull} + Dreamer baseline.
# Usage: ood_gp_matrix.sh ["axes"] ["seeds"]
set -uo pipefail
AXES=${1:-"color shape rotate"}; SEEDS=${2:-"0"}
LEWM=/home/seongbin/latent/le-wm; LCBF=/home/seongbin/latent/latent_cbf
DR=/data/seongbin/dreamer/dreamer; LOG=/home/seongbin/latent/run_logs
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
mkdir -p "$LOG"
for s in $SEEDS; do
  for axis in $AXES; do
    echo "[$(date +%H:%M:%S)] JEPA gp/nogp $axis seed=$s"
    CUDA_VISIBLE_DEVICES=0 "$LEWM/.venv/bin/python" -u "$LEWM/scripts/ood_margin_gp_jepa.py" \
      "$axis" "$s" > "$LOG/ood_gp_jepa_${axis}_s${s}.log" 2>&1
    grep "KEY jepa" "$LOG/ood_gp_jepa_${axis}_s${s}.log" || tail -3 "$LOG/ood_gp_jepa_${axis}_s${s}.log"
    echo "[$(date +%H:%M:%S)] Dreamer-baseline gp/nogp $axis seed=$s"
    CUDA_VISIBLE_DEVICES=0 "$LCBF/.venv/bin/python" -u \
      "$LCBF/src/latent_cbf/scripts/ood_margin_gp_dreamer.py" \
      --rssm_ckpt "$DR/enc_lip_sweep/baseline/rssm_ckpt.pt" --axis "$axis" --tag baseline --seed "$s" \
      > "$LOG/ood_gp_dreamer_${axis}_s${s}.log" 2>&1
    grep "KEY dreamer" "$LOG/ood_gp_dreamer_${axis}_s${s}.log" || tail -3 "$LOG/ood_gp_dreamer_${axis}_s${s}.log"
  done
done
echo OOD_GP_DONE
