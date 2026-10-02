#!/usr/bin/env bash
# Per-k h-fidelity sweep across encoder-reg variants:
# for each LeWM checkpoint, train an HJ-labeled margin head (goal-randomized,
# imagined-augmented), then measure corr(h(z_k), V*(x_k)) at k=1..25.
# Output: run_logs/perk_<variant>.log (one diag line per k) + trained heads.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf
export STABLEWM_HOME=/data/seongbin/lewm
PY=.venv/bin/python
LOG=/home/seongbin/latent/run_logs
GPU=${GPU:-1}

VARIANTS=(no_reg_dubins jacobian_w1_dubins jacobian_fd_w1_dubins state_lipschitz_w1_dubins pixel_lipschitz_dubins invariance_dubins)

for v in "${VARIANTS[@]}"; do
  head=/data/seongbin/dreamer/lewm/$v/margin_hj_imag.pt
  if [ ! -f "$head" ]; then
    echo "[$(date +%H:%M:%S)] train margin head: $v"
    CUDA_VISIBLE_DEVICES=$GPU $PY src/latent_cbf/scripts/train_margin_hj_lewm.py \
      --lewm_ckpt "$v/weights_epoch_50.pt" --out_path "$head" \
      --rollout_k 8 --steps 6000 --device cuda:0 \
      > "$LOG/train_margin_hj_$v.log" 2>&1 || { echo "TRAIN FAILED: $v"; continue; }
  fi
  echo "[$(date +%H:%M:%S)] per-k diag: $v"
  : > "$LOG/perk_$v.log"
  for k in 1 4 8 12 18 25; do
    CUDA_VISIBLE_DEVICES=$GPU $PY src/latent_cbf/scripts/diag_margin_on_imagined.py \
      --lewm_ckpt "$v/weights_epoch_50.pt" --margin_ckpt "$head" \
      --rollout_k $k --n 600 --device cuda:0 2>/dev/null | tail -1 >> "$LOG/perk_$v.log"
  done
done
echo "PERK_SWEEP_COMPLETE"
