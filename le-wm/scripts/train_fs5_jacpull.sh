#!/usr/bin/env bash
# Frameskip-5 jac+pull on Dubins (2026-11-08): each predictor step = 5 env steps (action encoder input = 5 stacked
# actions), so a 15-25-step goal is 3-5 predictor steps, inside the window where h still ranks imagined purple states.
# Recipe = jac (jacobian_w1_dubins) + pull exactly as the frameskip-5 pull rescue (lewm_dubins_fs5pull50: action_separation
# weight 0.1, pull_target 1.0), code snapshot code_safeadv_92e7589 (as the jac seeds), 50 epochs, seed 3072.
# Pull target = $2 (default 2.2: the measured free action ratio of the frameskip-1 Dubins WMs, as Reacher set its
# target from the measured 2.4; the July rescue used 1.0). Usage: train_fs5_jacpull.sh <gpu> [target]
set -uo pipefail
C=/data/seongbin/lewm/code_safeadv_92e7589; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
T=${2:-2.2}; n=lewm_dubins_fs5jacpull${T/./}_50
mkdir -p ".run_$n"; cd ".run_$n"
CUDA_VISIBLE_DEVICES=$1 $PY ../train.py trainer.max_epochs=50 seed=3072 wandb.enabled=false output_model_name=$n subdir=$n \
  data=dubins data.dataset.frameskip=5 \
  '+loss.regularizers.jacobian.weight=1.0' '+loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}' \
  '+loss.regularizers.action_separation.weight=0.1' "+loss.regularizers.action_separation.kwargs={pull_target:$T}" \
  > /home/seongbin/latent/run_logs/train_$n.log 2>&1
echo "exit $?"
