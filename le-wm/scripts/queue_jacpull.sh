#!/usr/bin/env bash
# Wait for >=28GB free on a GPU, then train jacobian+pull dubins combo.
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
echo "[$(date +%H:%M:%S)] waiting for >=28GB free GPU..."
GPU=""
while [ -z "$GPU" ]; do
  while read -r idx free; do
    if [ "$free" -ge 28000 ]; then GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  [ -z "$GPU" ] && sleep 120
done
echo "[$(date +%H:%M:%S)] GPU $GPU free, launching jacpull50 training"
mkdir -p .run_dubins_jacpull50
cd .run_dubins_jacpull50
CUDA_VISIBLE_DEVICES=$GPU ../.venv/bin/python ../train.py data=dubins trainer.max_epochs=50 \
  output_model_name=lewm_dubins_jacpull50 subdir=dubins_jacpull50 \
  +loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs.target_L=1.0 +loss.regularizers.jacobian.kwargs.n_probes=1 \
  +loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs.pull_target=1.0 \
  > "$LOG/train_dubins_jacpull50.log" 2>&1
echo "[$(date +%H:%M:%S)] jacpull50 training DONE"
echo JACPULL_TRAIN_DONE
