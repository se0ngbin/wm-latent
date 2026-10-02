#!/usr/bin/env bash
# jac+pull GRU (does the pull term tame carried memory / restore re-grounding?):
# GRU recurrent predictor + jacobian(1.0) + action_separation pull(0.1), sigreg.
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs
mkdir -p .run_gru_jacpull && cd .run_gru_jacpull
CUDA_VISIBLE_DEVICES=${GPU:-0} ../.venv/bin/python ../train.py data=dubins model=lewm_gru history_size=8 \
  trainer.max_epochs=50 output_model_name=lewm_gru_jacpull subdir=dubins_gru_jacpull \
  +loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs.target_L=1.0 +loss.regularizers.jacobian.kwargs.n_probes=1 \
  +loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs.pull_target=1.0 \
  > "$LOG/train_dubins_gru_jacpull.log" 2>&1
echo "GRU_JACPULL_TRAIN_DONE exit=$?"
