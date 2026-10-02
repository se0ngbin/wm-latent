#!/usr/bin/env bash
# JEPA+GRU (Dreamer-faithful memory ablation): baseline reg (sigreg only) + GRU
# recurrent predictor (deter_dim 512), history_size 8. get_feat=concat(embed,deter_h).
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs
mkdir -p .run_gru && cd .run_gru
CUDA_VISIBLE_DEVICES=${GPU:-0} ../.venv/bin/python ../train.py data=dubins model=lewm_gru history_size=8 \
  trainer.max_epochs=50 output_model_name=lewm_gru_dubins subdir=dubins_gru \
  > "$LOG/train_dubins_gru.log" 2>&1
echo "GRU_TRAIN_DONE exit=$?"
