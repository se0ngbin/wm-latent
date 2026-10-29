#!/usr/bin/env bash
# Encoder-term-only versions of every Dubins safety-reg WM (adversarial penalty on h(f(x+δ)) only, pred_weight=0): tests
# whether the predictor term is what costs planning (rollout error, goal geometry). adv_h's already exists
# (lewm_dubins_sa_enc50). Waits for train_dubins_jacadv.sh to finish, then launches each run once a GPU has >= 30 GB free.
# Same recipes / code snapshot as train_dubins_jacadv.sh plus pred_weight:0.0.
set -uo pipefail
[ "${WAIT_JACADV:-1}" = 1 ] && while pgrep -f train_dubins_jacadv.sh >/dev/null; do sleep 300; done   # WAIT_JACADV=0: start as soon as a GPU has room
C=/data/seongbin/lewm/code_safeadv_92e7589; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs={pull_target:1.0}'
SAW='+loss.regularizers.safety_adv.weight=1.0'
SA='+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725,pred_weight:0.0'
HEAD="${SA},auc_gate:0.95}"
LAM="${SA},auc_gate:0.0,target:lambda,lambda_mode:batch_detached}"
declare -A RUN=(
  [lewm_dubins_jac_safeadv_linf_enc50]="$JAC $SAW $HEAD"
  [lewm_dubins_jacpull_safeadv_linf_enc50]="$JAC $PULL $SAW $HEAD"
  [lewm_dubins_lam_bdet_enc50]="$SAW $LAM"
  [lewm_dubins_jac_lam_bdet_enc50]="$JAC $SAW $LAM"
  [lewm_dubins_jacpull_lam_bdet_enc50]="$JAC $PULL $SAW $LAM"
)
for name in "${@:-${!RUN[@]}}"; do
  [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt" ] && { echo "cached $name"; continue; }
  while :; do
    read -r gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ',')
    [ "$free" -ge 30000 ] && break; sleep 120
  done
  mkdir -p ".run_$name"
  # shellcheck disable=SC2086
  ( cd ".run_$name" && CUDA_VISIBLE_DEVICES=$gpu $PY ../train.py data=dubins_safety trainer.max_epochs=50 seed=3072 \
      wandb.enabled=false output_model_name=$name subdir=$name ${RUN[$name]} ) > "$LOG/train_$name.log" 2>&1 \
    && echo "$(date +%H:%M) done $name" || echo "$(date +%H:%M) FAILED $name (see $LOG/train_$name.log)" &
  sleep 240   # memory peaks in the first minutes
done
wait; echo DUBINS_ENC_ONLY_DONE
