#!/usr/bin/env bash
# jac+adv_h Dubins WM seeds 1 and 2 (never trained by train_seeds_dubins_rest.sh), same recipe and code snapshot as
# lewm_dubins_jac_safeadv_linf50 there. Seed 1 on GPU ${G1:-0}, seed 2 on GPU ${G2:-1}, in parallel, isolated cwds.
set -uo pipefail
C=/data/seongbin/lewm/code_safeadv_92e7589; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
LOG=/home/seongbin/latent/run_logs
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
SAW='+loss.regularizers.safety_adv.weight=1.0'
SA='+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725,auc_gate:0.95}'
train () {   # train <seed> <gpu>
  local name=lewm_dubins_jac_safeadv_linf50_s$1
  [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_50.pt" ] && { echo "cached $name"; return; }
  mkdir -p ".run_$name"
  # shellcheck disable=SC2086
  if ( cd ".run_$name" && CUDA_VISIBLE_DEVICES=$2 $PY ../train.py trainer.max_epochs=50 seed=$1 wandb.enabled=false \
         output_model_name=$name subdir=$name data=dubins_safety $JAC $SAW $SA ) > "$LOG/train_$name.log" 2>&1; then
    echo "$(date +%H:%M) done $name (gpu $2)"
  else
    echo "$(date +%H:%M) FAILED $name (gpu $2, see $LOG/train_$name.log)"
  fi
}
train 1 "${G1:-0}" &
train 2 "${G2:-1}" &
wait; echo JACADVH_SEEDS_DONE
