#!/usr/bin/env bash
# The 3 Dubins WMs missing from the 3x3 grid: jac+adv_h, jac+adv_λ, jac+pull+adv_λ. Same recipes as the Safety Gym / cartpole
# versions (code_point_92e7589/run_point.sh) with Dubins data; code snapshot code_safeadv_92e7589 (main's module.py lacks
# lambda_mode=batch_detached). Isolated cwd per run (wandb resume-file collision). Usage: train_dubins_jacadv.sh [name ...]
set -uo pipefail
C=/data/seongbin/lewm/code_safeadv_92e7589; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
LOG=/home/seongbin/latent/run_logs
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
PULL='+loss.regularizers.action_separation.weight=0.1 +loss.regularizers.action_separation.kwargs={pull_target:1.0}'
SAW='+loss.regularizers.safety_adv.weight=1.0'
SA='+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725'
LAM="${SA},auc_gate:0.0,target:lambda,lambda_mode:batch_detached}"
declare -A RUN=(
  [lewm_dubins_jac_safeadv_linf50]="$JAC $SAW ${SA},auc_gate:0.95}"
  [lewm_dubins_jac_lam_bdet50]="$JAC $SAW $LAM"
  [lewm_dubins_jacpull_lam_bdet50]="$JAC $PULL $SAW $LAM"
)
declare -A GPU=([lewm_dubins_jac_safeadv_linf50]=0 [lewm_dubins_jac_lam_bdet50]=1 [lewm_dubins_jacpull_lam_bdet50]=0)
for name in "${@:-${!RUN[@]}}"; do
  [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt" ] && { echo "cached $name"; continue; }
  mkdir -p ".run_$name"
  # shellcheck disable=SC2086
  ( cd ".run_$name" && CUDA_VISIBLE_DEVICES=${GPU[$name]} $PY ../train.py data=dubins_safety trainer.max_epochs=50 seed=3072 \
      wandb.enabled=false output_model_name=$name subdir=$name ${RUN[$name]} ) > "$LOG/train_$name.log" 2>&1 \
    && echo "$(date +%H:%M) done $name" || echo "$(date +%H:%M) FAILED $name (see $LOG/train_$name.log)" &
  sleep 60
done
wait; echo DUBINS_JACADV_DONE
