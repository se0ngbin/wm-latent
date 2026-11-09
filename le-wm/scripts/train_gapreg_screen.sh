#!/usr/bin/env bash
# 15-epoch screen (2026-11-07) of four gap-keeping terms on top of jac, plus a jac control at the same budget (cosine
# schedule annealed over the 15 epochs, so each is a complete short run). No augmentation, red data only.
#   jac_cons15     readout consistency (h(P(z_t,a_t)) - h(z_{t+1}))^2 / gap^2      safety_gap cons_weight
#   jac_predjac15  predictor Jacobian, last-frame gain pulled to 1 (replaces pull)   pred_jacobian
#   jac_encgap15   encoder gap hinge relu(0.5 - s h(z))                               safety_gap enc_gap_weight
#   jac_predgap15  predictor gap hinge on a 5-step free-running rollout                safety_gap pred_gap_weight
#   jac15          control
# Code: /data/seongbin/lewm/code_gapreg_1893547 (= le-wm/gapreg at commit 1893547). seed 3072.
# Screen next: diag_imag_color_dubins.py (h ranking on imagined purple latents per step) vs jac15.
set -uo pipefail
C=/data/seongbin/lewm/code_gapreg_1893547; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
LOG=/home/seongbin/latent/run_logs
JAC='+loss.regularizers.jacobian.weight=1.0 +loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}'
SG='+loss.regularizers.safety_gap.weight=1.0 +loss.regularizers.safety_gap.kwargs'
declare -A RUN=(
  [lewm_dubins_jac15]="data=dubins $JAC"
  [lewm_dubins_jac_cons15]="data=dubins_safety $JAC $SG={cons_weight:1.0}"
  [lewm_dubins_jac_predjac15]="data=dubins $JAC +loss.regularizers.pred_jacobian.weight=1.0"
  [lewm_dubins_jac_encgap15]="data=dubins_safety $JAC $SG={enc_gap_weight:1.0,gamma:0.5}"
  [lewm_dubins_jac_predgap15]="data=dubins_safety $JAC $SG={pred_gap_weight:1.0,gamma:0.5,rollout_steps:5}"
)
declare -A NEED=([lewm_dubins_jac_predgap15]=45000)
ORDER=(lewm_dubins_jac15 lewm_dubins_jac_encgap15 lewm_dubins_jac_cons15 lewm_dubins_jac_predjac15 lewm_dubins_jac_predgap15)
gpu_with_room () {   # prints a GPU id with >= $1 MiB free, waiting if none
  while :; do
    read -r g free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ',')
    [ "$free" -ge "$1" ] && { echo "$g"; return; }; sleep 120
  done
}
run () {
  local name=$1
  [ -f "$STABLEWM_HOME/checkpoints/$name/weights_epoch_15.pt" ] && { echo "cached $name"; return; }
  exec {fd}>"$LOG/.gapreg_gpu.lock"; flock "$fd"
  local g; g=$(gpu_with_room "${NEED[$name]:-30000}")
  mkdir -p ".run_$name"
  # shellcheck disable=SC2086
  ( cd ".run_$name" && CUDA_VISIBLE_DEVICES=$g $PY ../train.py trainer.max_epochs=15 seed=3072 wandb.enabled=false \
      output_model_name=$name subdir=$name ${RUN[$name]} ) > "$LOG/train_$name.log" 2>&1 &
  local pid=$!; echo "$(date +%H:%M) launched $name on gpu $g"; sleep 240; flock -u "$fd"; exec {fd}>&-
  wait $pid && echo "$(date +%H:%M) done $name" || echo "$(date +%H:%M) FAILED $name (see $LOG/train_$name.log)"
}
for n in "${@:-${ORDER[@]}}"; do run "$n" & sleep 5; done
wait; echo GAPREG_SCREEN_DONE
