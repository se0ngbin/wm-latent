#!/usr/bin/env bash
# LeWM WMs on CLEAN SafetyPointGoal1 pixels (sg_point_clean.h5: no cost-indicator sphere / lidar ring).
# Code = safety-adv-reg snapshot 92e7589; recipes copied from the Car-SG reference runs
# (train_sg.sh, run_sgLg0.sh, run_sgjpl_s3072.sh, run_sg_lam_bdet.sh) + jac/jacpull x {L-inf, lambda-bdet} combos.
# 5 runs on GPU0, 4 on GPU1, isolated cwd per run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PY:-/home/seongbin/latent/le-wm/.venv/bin/python}   # PY=echo for a dry run
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs/sg_point; mkdir -p "$LOG"
SEED=${SEED:-3072}; SFX=${SFX:-}
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs={pull_target:1.0}'
SAW='+loss.regularizers.safety_adv.weight=1.0'
SA='+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725'
train_one(){ local name=$1$SFX gpu=$2; shift 2
  [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt" ] && { echo "  cached $name"; return; }
  mkdir -p .run_$name; echo "[$(date +%H:%M:%S)] train $name (gpu $gpu)"
  ( cd .run_$name && CUDA_VISIBLE_DEVICES=$gpu $PY ../train.py data=sg_point_safety trainer.max_epochs=50 seed=$SEED \
      wandb.enabled=false output_model_name=$name subdir=$name "$@" ) > "$LOG/train_$name.log" 2>&1 \
    && echo "  done $name" || echo "  FAILED $name (see $LOG/train_$name.log)"; }
LAM="${SA},auc_gate:0.0,target:lambda,lambda_mode:batch_detached}"
declare -A RUN=(
  [pt_baseline]=""
  [pt_jacobian]="$JAC"
  [pt_jacpull]="$JAC $PULL"
  [pt_safeadv_linf_g0]="$SAW ${SA},auc_gate:0.0}"
  [pt_jacpull_safeadv_linf]="$JAC $PULL $SAW ${SA},auc_gate:0.95}"
  [pt_jac_safeadv_linf]="$JAC $SAW ${SA},auc_gate:0.95}"
  [pt_lam_bdet]="$SAW $LAM"
  [pt_jac_lam_bdet]="$JAC $SAW $LAM"
  [pt_jacpull_lam_bdet]="$JAC $PULL $SAW $LAM"
)   # no override contains spaces inside a single override, so word-splitting the strings is safe
# GPUs are shared with other jobs (e.g. cartpole runs): launch only when a GPU has >= NEED_MB free, settle 3 min
# between launches (a run's memory peaks during its first minutes). Pass run names as args to queue a subset.
NEED_MB=${NEED_MB:-30000}
for name in "${@:-${!RUN[@]}}"; do
  while :; do
    read -r gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ',')
    [ "$free" -ge "$NEED_MB" ] && break; sleep 60
  done
  # shellcheck disable=SC2086
  train_one "$name" "$gpu" ${RUN[$name]} &
  sleep 180
done
wait; echo "POINT_WM_DONE"
