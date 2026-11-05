#!/usr/bin/env bash
# Real-frame OOD eval of extra WM seeds: train the GP and the linear margin function (red, seed 0, if missing), then
# AUC / balanced sign acc / medians on red, purple, diamond, rot90 for both (eval_head_color_dubins.py).
# Usage: GPU=<id> seed_head_eval.sh <wm_dir> [...]
set -uo pipefail
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python; export CUDA_VISIBLE_DEVICES=${GPU:-1}
for ck in "$@"; do
  mkdir -p $M/$ck
  [ -f $M/$ck/margin_gp_fail_red_s0.pt ] || $PY train_gp_margin_dubins.py $ck 0 2>&1 | grep -E "saved|Error"
  [ -f $M/$ck/margin_lin_fail_red_s0.pt ] || $PY train_lin_margin_dubins.py $ck 0 2>&1 | grep -E "saved|Error"
  $PY eval_head_color_dubins.py $ck $M/$ck/margin_lin_fail_red_s0.pt $M/$ck/margin_gp_fail_red_s0.pt 2>&1 | grep -E "HEADOOD|Error"
done
echo SEED_HEAD_EVAL_DONE
