#!/usr/bin/env bash
# LEWM WMs on the cartpole dataset (dubins recipes, no color aug), same trio as train_sg.sh.
# Runs from the main-checkout le-wm (its uncommitted train.py has the action_separation reg).
# Per-run isolated cwd (wandb collision guard).
set -uo pipefail
cd /home/seongbin/latent/le-wm
PY=.venv/bin/python
: "${STABLEWM_HOME:=/data/seongbin/lewm}"; export STABLEWM_HOME
export LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs/cartpole; mkdir -p "$LOG"
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs.target_L=1.0 +loss.regularizers.jacobian.kwargs.n_probes=1'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs.pull_target=1.0'
train_one(){ local name=$1 gpu=$2; shift 2; local override="$*"
  local ck=$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt
  [ -f "$ck" ] && { echo "  cached $name"; return; }
  local rundir=.run_$name; mkdir -p "$rundir"
  echo "[$(date +%H:%M:%S)] train $name (gpu $gpu)"
  ( cd "$rundir" && CUDA_VISIBLE_DEVICES=$gpu ../$PY ../train.py data=cartpole trainer.max_epochs=50 \
      wandb.enabled=false output_model_name=$name subdir=$name hydra.run.dir=outputs/$name $override ) \
    > "$LOG/train_$name.log" 2>&1 && echo "  done $name" || echo "  FAILED $name (see $LOG/train_$name.log)"; }
( train_one cp_baseline 0 ""; train_one cp_jacpull 0 "$JAC $PULL" ) &
( train_one cp_jacobian 1 "$JAC" ) &
wait; echo "CP_WM_DONE"
