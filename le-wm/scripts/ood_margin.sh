#!/usr/bin/env bash
# OOD margin experiment: WM frozen, retrain the margin (safe/unsafe CBF classifier)
# for a shifted obstacle appearance, compare to the margin trained on the original
# appearance. Per encoder (baseline / jacobian / jacobian+pull).
#   axis=color : OG=red circle,  NEW=purple circle
#   axis=shape : OG=red circle,  NEW=red diamond (square)
# Usage: ood_margin.sh <color|shape>
set -uo pipefail
AXIS=${1:-color}
RK=${2:-6}     # rollout_k: 6 = imagined-latent training (deployed protocol); 0 = static
SEED=${3:-0}
FRESH=${4:-0}  # 1 = re-init the margin head (fair OOD oracle); 0 = pretrained init
FRESH_FLAG=""; [ "$FRESH" = 1 ] && FRESH_FLAG="--fresh_head"
LCBF=/home/seongbin/latent/latent_cbf
LEWM=/home/seongbin/latent/le-wm
MDIR=/data/seongbin/dreamer/lewm
LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm

declare -A ENC=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
NS=12000; STEPS=4000
SUF="_rk${RK}_s${SEED}"

if [ "$AXIS" = "color" ]; then
  OG_TAG=red;    OG_ARGS="--obstacle_color red"
  NEW_TAG=purple; NEW_ARGS="--obstacle_color purple"
elif [ "$AXIS" = "shape" ]; then
  OG_TAG=circle; OG_ARGS="--obstacle_shape circle"
  NEW_TAG=square; NEW_ARGS="--obstacle_shape square"
else  # rotate
  OG_TAG=rot0;  OG_ARGS="--rotate_deg 0"
  NEW_TAG=rot90; NEW_ARGS="--rotate_deg 90"
fi

train () {  # ck tag args
  local ck=$1 tag=$2 args=$3
  local out="$MDIR/$ck/margin_hj_${AXIS}_${tag}${SUF}.pt"
  echo "[$(date +%H:%M:%S)] train margin $ck $AXIS=$tag rk=$RK seed=$SEED"
  CUDA_VISIBLE_DEVICES=0 "$LCBF/.venv/bin/python" -u \
    "$LCBF/src/latent_cbf/scripts/train_margin_hj_lewm.py" \
    --lewm_ckpt "$ck/weights_epoch_50.pt" --out_path "$out" \
    $args $FRESH_FLAG --n_states $NS --rollout_k $RK --steps $STEPS --seed $SEED \
    > "$LOG/ood_margin_train_${AXIS}_${ck}_${tag}${SUF}.log" 2>&1
  grep -E "val_sign_acc|saved" "$LOG/ood_margin_train_${AXIS}_${ck}_${tag}${SUF}.log" | tail -2
}

for name in "${!ENC[@]}"; do
  ck=${ENC[$name]}
  train "$ck" "$OG_TAG"  "$OG_ARGS"
  train "$ck" "$NEW_TAG" "$NEW_ARGS"
done

echo "[$(date +%H:%M:%S)] eval axis=$AXIS rk=$RK seed=$SEED"
cd "$LEWM"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/ood_margin_eval.py "$AXIS" "$RK" "$SEED" \
  > "$LOG/ood_margin_eval_${AXIS}${SUF}.log" 2>&1
grep -vE "Warning|INFO|Gym|migration|ale" "$LOG/ood_margin_eval_${AXIS}${SUF}.log"
echo "OOD_MARGIN_${AXIS}${SUF}_DONE"
