#!/usr/bin/env bash
# jac_predgap15 launched directly (2026-11-07 10:45 PST) instead of waiting for 45 GB in train_gapreg_screen.sh; same
# command and code snapshot. Usage: train_gapreg_predgap_now.sh <gpu>
set -uo pipefail
C=/data/seongbin/lewm/code_gapreg_1893547; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
n=lewm_dubins_jac_predgap15
mkdir -p ".run_$n"; cd ".run_$n"
CUDA_VISIBLE_DEVICES=$1 $PY ../train.py trainer.max_epochs=15 seed=3072 wandb.enabled=false output_model_name=$n subdir=$n \
  data=dubins_safety '+loss.regularizers.jacobian.weight=1.0' '+loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}' \
  '+loss.regularizers.safety_gap.weight=1.0' '+loss.regularizers.safety_gap.kwargs={pred_gap_weight:1.0,gamma:0.5,rollout_steps:5}' \
  > /home/seongbin/latent/run_logs/train_$n.log 2>&1
echo "exit $?"
