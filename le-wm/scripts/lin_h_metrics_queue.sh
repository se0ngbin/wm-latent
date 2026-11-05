#!/usr/bin/env bash
# Train a linear margin function (if missing) for each WM, then real-frame AUC / balanced sign acc of linear AND GP heads
# on red / purple / diamond / rot90 (eval_head_color_dubins.py). Usage: GPU=<id> lin_h_metrics_queue.sh <wm_dir> [...]
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=/home/seongbin/latent/le-wm/.venv/bin/python; H=/data/seongbin/dreamer/lewm
for ck in "$@"; do
  [ -f $H/$ck/margin_lin_fail_red_s0.pt ] || $PY train_lin_margin_dubins.py $ck 0
  $PY eval_head_color_dubins.py $ck $H/$ck/margin_lin_fail_red_s0.pt $H/$ck/margin_gp_fail_red_s0.pt 2>&1 | grep -E "HEADOOD|Error"
done
echo LIN_METRICS_DONE
