#!/usr/bin/env bash
# Dreamer GT-margin value ablation (Post Sept 16). Same predictor-isolation as the
# LEWM version: train the reachability critic with CRITIC_GT_MARGIN=1 (reward = true
# signed distance, tracked from privileged_state along the latent rollout) on the
# Dreamer/RSSM backend, then eval value AUC vs HJ V* on red/purple/diamond/rot90.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
GPU=${GPU:-1}
LOG=/home/seongbin/latent/run_logs/gt_margin_value; mkdir -p "$LOG"
VDIR=/data/seongbin/dreamer/lewm/_gt_margin_value; mkdir -p "$VDIR"
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
logdir=$VDIR/dreamer

pol=$(ls -d "$logdir"/PyHJ/gp/epoch_id_* 2>/dev/null | head -1)
if [ -n "$pol" ] && [ -f "$pol/policy.pth" ]; then
  echo "  cached dreamer"
else
  echo "[$(date +%H:%M:%S)] train GT-margin critic dreamer (gpu $GPU)"
  CRITIC_GT_MARGIN=1 CUDA_VISIBLE_DEVICES=$GPU $PY scripts/wm_ddpg.py --wm_backend dreamer \
    --rssm-ckpt "$RSSM" --logdir "$logdir" --seed 0 > "$LOG/train_dreamer.log" 2>&1 \
    && echo "  done dreamer" || { echo "  FAILED dreamer"; tail -5 "$LOG/train_dreamer.log"; exit 1; }
fi

echo "[$(date +%H:%M:%S)] eval value AUC vs HJ V*"
pol=$(ls -d "$logdir"/PyHJ/gp/epoch_id_* 2>/dev/null | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/critic_ood_eval_dreamer.py --tag "gtmargin_dreamer" \
  --rssm_ckpt "$RSSM" --policy "$pol" > "$LOG/eval_dreamer.log" 2>&1
grep "^KEY critic" "$LOG/eval_dreamer.log"
echo "GT_MARGIN_VALUE_DREAMER_DONE"
