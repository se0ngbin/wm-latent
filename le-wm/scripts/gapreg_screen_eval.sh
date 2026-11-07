#!/usr/bin/env bash
# Screen for the 15-epoch gap-keeping runs (train_gapreg_screen.sh): as each weights_epoch_15.pt lands, alias it as
# <name>_e50 (scripts load weights_epoch_50.pt), train the MLP and linear margin functions (red uniform states, the
# usual recipe), then diag_imag_color_dubins.py: AUC of h on imagined latents per step from a red and a purple start.
# Reference: the 50-epoch jac (jacobian_w1_dubins). Output: run_logs/gapreg_screen_eval.log (IMAGCOLOR lines).
# Usage: gapreg_screen_eval.sh <gpu> [name ...]
set -uo pipefail
cd "$(dirname "$0")"
G=$1; shift
PY=/home/seongbin/latent/le-wm/.venv/bin/python
CK=/data/seongbin/lewm/checkpoints; M=/data/seongbin/dreamer/lewm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$G
screen () {   # wm_dir (as load_pretrained sees it)
  local w=$1
  for h in gp lin; do
    [ -f $M/$w/margin_${h}_fail_red_s0.pt ] && continue
    if [ $h = gp ]; then $PY train_gp_margin_dubins.py $w 0; else $PY train_lin_margin_dubins.py $w 0; fi 2>&1 | grep -E "saved|Error"
  done
  $PY diag_imag_color_dubins.py $w $M/$w/margin_gp_fail_red_s0.pt $M/$w/margin_lin_fail_red_s0.pt 2>&1 | grep -E "IMAGCOLOR|Error"
}
for n in "$@"; do
  until [ -f $CK/$n/weights_epoch_15.pt ]; do sleep 120; done
  sleep 30   # let the save finish
  a=${n}_e50
  if [ ! -d $CK/$a ]; then
    mkdir -p $CK/$a
    ln -s $CK/$n/config.json $CK/$a/config.json; ln -s $CK/$n/config.yaml $CK/$a/config.yaml
    ln -s $CK/$n/weights_epoch_15.pt $CK/$a/weights_epoch_50.pt
  fi
  screen $a
done
echo GAPREG_SCREEN_EVAL_DONE
