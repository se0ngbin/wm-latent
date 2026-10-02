#!/usr/bin/env bash
# PREDICTOR-ISOLATION ablation for the value function (Post Sept 16).
# Train the reachability critic with the GROUND-TRUTH margin as reward
# (CRITIC_GT_MARGIN=1: track true (x,y,θ) alongside the latent rollout, reward =
# tanh signed-distance) instead of the learned margin head. Everything else is the
# canonical in-dist critic (buffer reset, learned predictor dynamics + latent
# features). Then eval value AUC vs HJ V* on red/purple/diamond/rot90. If the value
# fn still collapses OOD with a PERFECT reward, the fragility is the PREDICTOR, not
# the margin; compare the AUC to displacement + cos_la geometry.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs/gt_margin_value; mkdir -p "$LOG"
VDIR=/data/seongbin/dreamer/lewm/_gt_margin_value; mkdir -p "$VDIR"
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )

train_one() {  # tag gpu
  local tag=$1 gpu=$2; local w=${WM[$tag]}
  local ckpt=/data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt
  local margin=/data/seongbin/dreamer/lewm/$w/margin_heads.pt
  local logdir=$VDIR/$tag
  local pol; pol=$(ls -d "$logdir"/PyHJ/gp/epoch_id_* 2>/dev/null | head -1)
  if [ -n "$pol" ] && [ -f "$pol/policy.pth" ]; then echo "  cached $tag"; return; fi
  echo "[$(date +%H:%M:%S)] train GT-margin critic $tag (gpu $gpu)"
  CRITIC_GT_MARGIN=1 CUDA_VISIBLE_DEVICES=$gpu $PY scripts/wm_ddpg.py --wm_backend lewm \
    --lewm_run_name "$w" --lewm_ckpt_path "$ckpt" --lewm_margin_ckpt "$margin" \
    --logdir "$logdir" --seed 0 > "$LOG/train_$tag.log" 2>&1 \
    && echo "  done $tag" || echo "  FAILED $tag"
}

# train baseline+jacobian in parallel (gpu 0/1), then jacpull on 0
train_one baseline 0 &
train_one jacobian 1 &
wait
train_one jacpull 0

echo "[$(date +%H:%M:%S)] eval value AUC vs HJ V*"
for tag in baseline jacobian jacpull; do
  w=${WM[$tag]}; ckpt=/data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt
  margin=/data/seongbin/dreamer/lewm/$w/margin_heads.pt
  pol=$(ls -d "$VDIR/$tag"/PyHJ/gp/epoch_id_* 2>/dev/null | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth
  CUDA_VISIBLE_DEVICES=0 $PY scripts/critic_ood_eval.py --tag "gtmargin_$tag" \
    --lewm_ckpt "$ckpt" --margin_ckpt "$margin" --policy "$pol" \
    > "$LOG/eval_$tag.log" 2>&1
  grep "^KEY critic" "$LOG/eval_$tag.log"
done
echo "GT_MARGIN_VALUE_DONE"
