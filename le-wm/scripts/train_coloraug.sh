#!/usr/bin/env bash
# Obstacle color-aug WM training (Post Sept 16): recolor the red obstacle to a random
# non-purple training color per window (+color_aug=true), purple held OUT so it stays
# OOD. Three LEWM recipes matching the study's encoders. Each run in its own cwd
# (wandb_resume.json cwd-collision guard). 50 epochs, data=dubins.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
: "${STABLEWM_HOME:=/data/seongbin/lewm}"; export STABLEWM_HOME
export LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs/coloraug; mkdir -p "$LOG"

# name -> reg override (baseline = sigreg only)
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs.target_L=1.0 +loss.regularizers.jacobian.kwargs.n_probes=1'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs.pull_target=1.0'

train_one() {  # name gpu override...
  local name=$1 gpu=$2; shift 2; local override="$*"
  local ck=$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt
  [ -f "$ck" ] && { echo "  cached $name"; return; }
  local rundir=.run_$name; mkdir -p "$rundir"
  echo "[$(date +%H:%M:%S)] train $name (gpu $gpu)  override='$override'"
  ( cd "$rundir" && CUDA_VISIBLE_DEVICES=$gpu ../$PY ../train.py data=dubins trainer.max_epochs=50 \
      wandb.enabled=false output_model_name=$name subdir=$name hydra.run.dir=outputs/$name \
      +color_aug=true $override ) > "$LOG/train_$name.log" 2>&1 \
    && echo "  done $name" || echo "  FAILED $name (see $LOG/train_$name.log)"
}

# GPU 0: baseline then jacpull ; GPU 1: jacobian   (2 GPUs, 3 runs)
( train_one coloraug_baseline 0 ""; train_one coloraug_jacpull 0 "$JAC $PULL" ) &
( train_one coloraug_jacobian 1 "$JAC" ) &
wait
echo "COLORAUG_WM_DONE"
