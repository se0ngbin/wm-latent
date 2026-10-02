#!/usr/bin/env bash
# Dreamer baseline WM with obstacle color-aug (Post Sept 16). Same recipe as the
# enc_lip_sweep baseline RSSM (--enc_lip_weight 0.0) plus --color_aug (recolor the red
# obstacle to random non-purple colors per sequence; purple held out). Waits for a GPU
# with enough free memory (the LEWM color-aug trio is running), then trains.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs/coloraug; mkdir -p "$LOG"
OUT=/data/seongbin/dreamer/dreamer/enc_lip_sweep/coloraug_baseline

if [ -f "$OUT/rssm_ckpt.pt" ]; then echo "  cached dreamer coloraug"; exit 0; fi
GPU=${GPU:-1}
echo "[$(date +%H:%M:%S)] GPU $GPU -> train dreamer color-aug -> $OUT"

CUDA_VISIBLE_DEVICES=$GPU $PY scripts/dreamer_offline.py --steps 40000 \
  --enc_lip_weight 0.0 --color_aug --logdir "$OUT" > "$LOG/train_dreamer_coloraug.log" 2>&1 \
  && echo "  done dreamer coloraug" || { echo "  FAILED dreamer coloraug"; tail -8 "$LOG/train_dreamer_coloraug.log"; exit 1; }
echo "COLORAUG_DREAMER_WM_DONE"
