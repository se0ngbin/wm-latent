#!/usr/bin/env bash
# AC-MTM (contrastive inverse-dynamics anti-collapse, sigreg OFF) WMs, +/- jacobian, on dubins
# and safety-gym CarGoal. Same recipe as the existing baselines (50 ep, fs=1, bs 128).
# Seed 3072 first (matches the single-seed sigreg_only/jacobian_w1/jacpull50 and sg_* ckpts),
# then seeds 1, 2. Two lanes = one run per GPU. Per-run isolated cwd (wandb collision guard).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/seongbin/latent/le-wm/.venv/bin/python
: "${STABLEWM_HOME:=/data/seongbin/lewm}"; export STABLEWM_HOME
export LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs/acmtm; mkdir -p "$LOG"
ACMTM="~loss.regularizers.sigreg +loss.regularizers.action_nce.weight=0.3 +loss.regularizers.action_nce.kwargs.temperature=0.1"
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs.target_L=1.0 +loss.regularizers.jacobian.kwargs.n_probes=1'
train_one(){ local name=$1 gpu=$2 data=$3 seed=$4; shift 4; local override="$*"
  local ck=$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt
  [ -f "$ck" ] && { echo "  cached $name"; return; }
  local rundir=.run_$name; mkdir -p "$rundir"
  echo "[$(date +%F' '%H:%M:%S)] train $name (gpu $gpu)"
  ( cd "$rundir" && CUDA_VISIBLE_DEVICES=$gpu $PY ../train.py data=$data trainer.max_epochs=50 seed=$seed \
      wandb.enabled=false output_model_name=$name subdir=$name hydra.run.dir=outputs/$name $override ) \
    > "$LOG/train_$name.log" 2>&1 && echo "  done $name" || echo "  FAILED $name (see $LOG/train_$name.log)"; }
sfx(){ [ "$1" = 3072 ] && echo "" || echo "_s$1"; }
lane(){ local gpu=$1 jac=$2 tag=$3
  for seed in 3072 1 2; do
    for env in dubins sg; do
      local data=$([ $env = sg ] && echo sg_cargoal || echo dubins)
      train_one ${env}_${tag}$(sfx $seed) $gpu $data $seed "$ACMTM $jac"
    done
  done; }
lane 0 "" acmtm &
lane 1 "$JAC" acmtm_jac &
wait; echo ACMTM_WM_DONE
