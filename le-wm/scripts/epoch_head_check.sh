#!/usr/bin/env bash
# Score a WM at intermediate epochs as soon as each checkpoint lands: GP and linear margin functions (red-trained),
# real-frame AUC / sign acc on red, purple, diamond, rot90 (eval_head_color_dubins.py) and per-step AUC on imagined
# latents from red / purple starts (diag_imag_color_dubins.py), for both heads. The scripts load
# <ckpt>/weights_epoch_50.pt, so each epoch gets a symlink dir <SRC>_e<E>. Usage: GPU=<id> epoch_head_check.sh <SRC> <epoch> [...]
set -uo pipefail
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
C=/data/seongbin/lewm/checkpoints; M=/data/seongbin/dreamer/lewm
PY=/home/seongbin/latent/le-wm/.venv/bin/python; export CUDA_VISIBLE_DEVICES=${GPU:-0}
SRC=$1; shift
for E in "$@"; do
  src=$C/$SRC/weights_epoch_$E.pt; ck=${SRC}_e$E
  until [ -f $src ]; do sleep 120; done; sleep 60   # let the save finish
  mkdir -p $C/$ck $M/$ck; ln -sf $src $C/$ck/weights_epoch_50.pt
  ln -sf $C/$SRC/config.json $C/$ck/config.json; ln -sf $C/$SRC/config.yaml $C/$ck/config.yaml
  [ -f $M/$ck/margin_gp_fail_red_s0.pt ] || $PY train_gp_margin_dubins.py $ck 0 2>&1 | grep -E "saved|Error"
  [ -f $M/$ck/margin_lin_fail_red_s0.pt ] || $PY train_lin_margin_dubins.py $ck 0 2>&1 | grep -E "saved|Error"
  $PY eval_head_color_dubins.py $ck $M/$ck/margin_lin_fail_red_s0.pt $M/$ck/margin_gp_fail_red_s0.pt 2>&1 | grep -E "HEADOOD|Error"
  $PY diag_imag_color_dubins.py $ck $M/$ck/margin_lin_fail_red_s0.pt $M/$ck/margin_gp_fail_red_s0.pt 2>&1 | grep -E "IMAGCOLOR|Error"
done
echo "EPOCH_CHECK_DONE $SRC"
