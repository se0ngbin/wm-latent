#!/usr/bin/env bash
# Value-function OOD, RETRAINED column: train a reachability critic per (model,
# appearance) with (a) the per-appearance margin as reward and (b) reset-seed
# images re-rendered under that appearance (CRITIC_OBST_* -> dubins-wm-dp.py).
# 3 JEPA models x {purple, diamond, rot90}. ~45-60 min/critic, 2 at a time on 2 GPUs.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
VDIR=/data/seongbin/dreamer/lewm/_value_ood; MDIR=$VDIR/margins; LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
declare -A ENVV=( [purple]="CRITIC_OBST_COLOR=purple" [diamond]="CRITIC_OBST_SHAPE=diamond" [rot90]="CRITIC_ROT_DEG=90" )

train_one() {  # tag cond gpu
  local tag=$1 cond=$2 gpu=$3 w=${WM[$1]}
  local margin="$MDIR/${tag}_${cond}.pt"
  [ -f "$margin" ] || { echo "  MISSING margin $margin"; return 1; }
  echo "[$(date +%H:%M:%S)] train critic $tag/$cond (gpu $gpu, env ${ENVV[$cond]})"
  env ${ENVV[$cond]} CUDA_VISIBLE_DEVICES=$gpu $PY scripts/wm_ddpg.py --wm_backend lewm \
    --lewm_run_name "$w" --lewm_ckpt_path /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
    --lewm_margin_ckpt "$margin" --logdir "$VDIR/jepa_${tag}_${cond}" --seed 0 \
    > "$LOG/critic_retrained_jepa_${tag}_${cond}.log" 2>&1 \
    && echo "  done $tag/$cond" || echo "  FAILED $tag/$cond (exit $?)"
}

JOBS=(baseline:purple baseline:diamond baseline:rot90 \
      jacobian:purple jacobian:diamond jacobian:rot90 \
      jacpull:purple  jacpull:diamond  jacpull:rot90)
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  a=${JOBS[$i]}; b=${JOBS[$((i+1))]:-}
  train_one "${a%:*}" "${a#*:}" 0 &
  [ -n "$b" ] && train_one "${b%:*}" "${b#*:}" 1 &
  wait
  i=$((i+2))
done
echo CRITIC_RETRAINED_DONE
