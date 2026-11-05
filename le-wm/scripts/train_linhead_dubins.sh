#!/usr/bin/env bash
# adv_h with a LINEAR internal safety head (head_layers:0; the adversary then attacks h(z) = w.z + b, matching a linear
# margin function downstream). Same recipes as train_dubins_jacadv.sh / train_dubins_enc_only.sh otherwise, code snapshot
# code_safeadv_92e7589_linhead (= code_safeadv_92e7589 + the head_layers option). Usage: GPU=<id> SEED=<s> train_linhead_dubins.sh name [...]
set -uo pipefail
C=/data/seongbin/lewm/code_safeadv_92e7589_linhead; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
LOG=/home/seongbin/latent/run_logs; S=${SEED:-3072}
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs={pull_target:1.0}'
SAW='+loss.regularizers.safety_adv.weight=1.0'
SA='+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725,head_layers:0'
declare -A RUN=(
  [lewm_dubins_jac_safeadv_linf_linhead50]="$JAC $SAW ${SA},auc_gate:0.95}"
  [lewm_dubins_jacpull_safeadv_linf_enc_linhead50]="$JAC $PULL $SAW ${SA},pred_weight:0.0,auc_gate:0.95}"
  [lewm_dubins_jac_safeadv_linf_enc_linhead50]="$JAC $SAW ${SA},pred_weight:0.0,auc_gate:0.95}"
  [lewm_dubins_jacpull_safeadv_linf_linhead50]="$JAC $PULL $SAW ${SA},auc_gate:0.95}"
)
for base in "$@"; do
  name=$base; [ "$S" != 3072 ] && name=${base}_s$S
  [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt" ] && { echo "cached $name"; continue; }
  mkdir -p ".run_$name"
  # shellcheck disable=SC2086
  ( cd ".run_$name" && CUDA_VISIBLE_DEVICES=${GPU:?} $PY ../train.py data=dubins_safety trainer.max_epochs=50 seed=$S \
      wandb.enabled=false output_model_name=$name subdir=$name ${RUN[$base]} ) > "$LOG/train_$name.log" 2>&1 \
    && echo "$(date +%H:%M) done $name" || echo "$(date +%H:%M) FAILED $name (see $LOG/train_$name.log)" &
  sleep 20
done
wait; echo LINHEAD_DONE
