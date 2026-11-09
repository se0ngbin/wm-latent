#!/usr/bin/env bash
# WM seeds 1 and 2 for the 6 core Dubins models that only have seed 3072, with the ORIGINAL recipes and code snapshot
# (code_safeadv_92e7589, as train_dubins_jacadv.sh / train_dubins_enc_only.sh; jac as jacobian_w1_dubins/config.yaml).
# One lane per model (seeds sequential); lanes start in priority order and each run waits for a GPU with >= 30 GB free.
# Names follow the _s<seed> convention. Usage: train_seeds_dubins_rest.sh [name ...]
set -uo pipefail
C=/data/seongbin/lewm/code_safeadv_92e7589; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
LOG=/home/seongbin/latent/run_logs
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs={pull_target:1.0}'
SAW='+loss.regularizers.safety_adv.weight=1.0'
SA='+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725'
LAM="${SA},auc_gate:0.0,target:lambda,lambda_mode:batch_detached}"
LAM_ENC="${SA},pred_weight:0.0,auc_gate:0.0,target:lambda,lambda_mode:batch_detached}"
ORDER=(jacobian_w1_dubins lewm_dubins_jac_lam_bdet50 lewm_dubins_jacpull_lam_bdet50 lewm_dubins_jac_safeadv_linf50
       lewm_dubins_jac_lam_bdet_enc50 lewm_dubins_jacpull_lam_bdet_enc50)
declare -A RUN=(
  [jacobian_w1_dubins]="data=dubins $JAC"
  [lewm_dubins_jac_lam_bdet50]="data=dubins_safety $JAC $SAW $LAM"
  [lewm_dubins_jacpull_lam_bdet50]="data=dubins_safety $JAC $PULL $SAW $LAM"
  [lewm_dubins_jac_safeadv_linf50]="data=dubins_safety $JAC $SAW ${SA},auc_gate:0.95}"
  [lewm_dubins_jac_lam_bdet_enc50]="data=dubins_safety $JAC $SAW $LAM_ENC"
  [lewm_dubins_jacpull_lam_bdet_enc50]="data=dubins_safety $JAC $PULL $SAW $LAM_ENC"
)
gpu_with_room () {   # prints a GPU id with >= 30 GB free, waiting if none
  while :; do
    read -r g free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ',')
    [ "$free" -ge 30000 ] && { echo "$g"; return; }; sleep 120
  done
}
lane () {
  local base=$1
  for s in 1 2; do
    local name=${base}_s$s
    [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt" ] && { echo "cached $name"; continue; }
    # one launch at a time across lanes: hold the lock until 4 min after launch (memory peaks in the first minutes)
    exec {fd}>"$LOG/.seeds_rest_gpu.lock"; flock "$fd"
    local g; g=$(gpu_with_room)
    mkdir -p ".run_$name"
    # shellcheck disable=SC2086
    ( cd ".run_$name" && CUDA_VISIBLE_DEVICES=$g $PY ../train.py trainer.max_epochs=50 seed=$s wandb.enabled=false \
        output_model_name=$name subdir=$name ${RUN[$base]} ) > "$LOG/train_$name.log" 2>&1 &
    local pid=$!; sleep 240; flock -u "$fd"; exec {fd}>&-
    wait $pid && echo "$(date +%H:%M) done $name (gpu $g)" || echo "$(date +%H:%M) FAILED $name (see $LOG/train_$name.log)"
  done
}
for base in "${@:-${ORDER[@]}}"; do lane "$base" & sleep 5; done
wait; echo SEEDS_REST_DONE
