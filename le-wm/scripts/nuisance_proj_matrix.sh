#!/usr/bin/env bash
# Lever #3: nuisance-subspace projection at readout + safe/unsafe separability.
# 4 encoders. Projection focuses on color (the failure axis); separability is
# reported per encoder (axis-independent, uses in-dist features).
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs; OUT="$LOG/nuisance_proj.txt"; : > "$OUT"
CK=/data/seongbin/lewm/checkpoints; MG=/data/seongbin/dreamer/lewm
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
AXIS=${1:-color}
for tag in baseline jacobian jacpull; do
  w=${WM[$tag]}
  CUDA_VISIBLE_DEVICES=0 $PY scripts/nuisance_proj_eval.py --backend lewm --tag "$tag" \
    --enc "$CK/$w/weights_epoch_50.pt" --margin "$MG/$w/margin_heads.pt" --axis "$AXIS" \
    2>/dev/null | grep -E "^(SEP|PROJ)" | tee -a "$OUT"
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/nuisance_proj_eval.py --backend dreamer --tag dreamer \
  --rssm "$RSSM" --axis "$AXIS" 2>/dev/null | grep -E "^(SEP|PROJ)" | tee -a "$OUT"
echo NUISANCE_PROJ_DONE | tee -a "$OUT"
