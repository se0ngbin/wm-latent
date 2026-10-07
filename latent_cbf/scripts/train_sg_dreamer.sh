#!/usr/bin/env bash
# Dreamer baseline WM on the safety-gym CarGoal dataset (matches dubins dreamer recipe:
# enc_lip_weight 0). Waits for a GPU with >=40GB free (LEWM trio finishing), then trains.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs/sg; OUT=/data/seongbin/dreamer/dreamer/sg_baseline
[ -f "$OUT/rssm_ckpt.pt" ] && { echo "cached sg dreamer"; exit 0; }
GPU=""
while [ -z "$GPU" ]; do
  while read -r idx free; do [ "$free" -ge 85000 ] && { GPU=$idx; break; }; done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  [ -z "$GPU" ] && { echo "[$(date +%H:%M:%S)] waiting for free GPU..."; sleep 180; }
done
echo "[$(date +%H:%M:%S)] GPU $GPU -> train sg dreamer baseline"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU $PY scripts/dreamer_offline.py --steps 40000 \
  --enc_lip_weight 0.0 --action_dim 2 --batch_length 8 \
  --dataset_path /data/seongbin/lewm/datasets/sg_cargoal_dreamer.h5 --logdir "$OUT" \
  > "$LOG/train_sg_dreamer.log" 2>&1 && echo "done sg dreamer" || echo "FAILED sg dreamer"
