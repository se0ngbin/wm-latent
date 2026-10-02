#!/usr/bin/env bash
# Dreamer-only slice of coloraug_downstream (its WM is done first). Margin OOD AUC
# (3 axes) + value OOD AUC (critic on the color-aug RSSM). GPU via env (default 0).
set -uo pipefail
LCBF=/home/seongbin/latent/latent_cbf
DR=/data/seongbin/dreamer/dreamer; RSSM=$DR/enc_lip_sweep/coloraug_baseline/rssm_ckpt.pt
LOG=/home/seongbin/latent/run_logs/coloraug; mkdir -p "$LOG"; CPY=$LCBF/.venv/bin/python
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
GPU=${GPU:-0}; cd "$LCBF/src/latent_cbf"

echo "[$(date +%H:%M:%S)] dreamer margin OOD AUC"
for axis in color shape rotate; do
  CUDA_VISIBLE_DEVICES=$GPU "$CPY" -u scripts/ood_margin_gp_dreamer.py \
    --rssm_ckpt "$RSSM" --axis "$axis" --tag coloraug --seed 0 > "$LOG/margin_coloraug_dreamer_${axis}.log" 2>&1
  grep "KEY dreamer" "$LOG/margin_coloraug_dreamer_${axis}.log"
done

echo "[$(date +%H:%M:%S)] dreamer critic + value OOD AUC"
dlog=$DR/enc_lip_sweep/coloraug_baseline/critic_seed0
dpol=$(ls -d "$dlog"/PyHJ/gp/epoch_id_* 2>/dev/null | head -1)
[ -n "$dpol" ] && [ -f "$dpol/policy.pth" ] || CUDA_VISIBLE_DEVICES=$GPU "$CPY" scripts/wm_ddpg.py \
  --wm_backend dreamer --rssm-ckpt "$RSSM" --logdir "$dlog" --seed 0 > "$LOG/critic_dreamer.log" 2>&1
dpol=$(ls -d "$dlog"/PyHJ/gp/epoch_id_* | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth
CUDA_VISIBLE_DEVICES=$GPU "$CPY" scripts/critic_ood_eval_dreamer.py --tag "coloraug_dreamer" \
  --rssm_ckpt "$RSSM" --policy "$dpol" > "$LOG/value_dreamer.log" 2>&1
grep "^KEY critic" "$LOG/value_dreamer.log"
echo "COLORAUG_DREAMER_DOWNSTREAM_DONE"
