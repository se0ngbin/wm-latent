#!/usr/bin/env bash
# Paired appearance-shift latent probe: 4 encoders x 3 axes.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs; OUT="$LOG/latent_probe.txt"; : > "$OUT"
CD=/home/seongbin/latent/run_logs/probe_coords; mkdir -p "$CD"
CK=/data/seongbin/lewm/checkpoints; MG=/data/seongbin/dreamer/lewm
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
for axis in color shape rotate; do
  for tag in baseline jacobian jacpull; do
    w=${WM[$tag]}
    CUDA_VISIBLE_DEVICES=0 $PY scripts/latent_shift_probe.py --backend lewm --tag "$tag" \
      --enc "$CK/$w/weights_epoch_50.pt" --margin "$MG/$w/margin_heads.pt" --axis "$axis" \
      --dump "$CD/${tag}_${axis}.npz" 2>/dev/null | grep "KEY probe" | tee -a "$OUT"
  done
  CUDA_VISIBLE_DEVICES=0 $PY scripts/latent_shift_probe.py --backend dreamer --tag dreamer \
    --rssm "$RSSM" --axis "$axis" --dump "$CD/dreamer_${axis}.npz" 2>/dev/null | grep "KEY probe" | tee -a "$OUT"
done
echo LATENT_PROBE_DONE | tee -a "$OUT"
