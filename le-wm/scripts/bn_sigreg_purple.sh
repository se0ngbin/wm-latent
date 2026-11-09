#!/usr/bin/env bash
# Dubins sigreg (LeWM default, 3 seeds) with label-free test-time BN adaptation on 500 SAFE purple frames (the Reacher
# result: BN adaptation rescued sigreg there), CCEM minh, crossing goals, K = 15 / 20 / 25, 100 episodes, purple,
# threshold 0, linear + MLP h (red-trained), each paired with the same model unadapted.
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
lane () {   # gpu wm tag
  local g=$1 w=$2 t=$3 a=${2}_bnpurple_safe500
  for h in lin gp; do [ -f $M/$w/margin_${h}_fail_red_s0.pt ] || { [ $h = gp ] && CUDA_VISIBLE_DEVICES=$g $PY train_gp_margin_dubins.py $w 0 || CUDA_VISIBLE_DEVICES=$g $PY train_lin_margin_dubins.py $w 0; } 2>&1 | grep -E "saved|Error"; done
  [ -f $STABLEWM_HOME/checkpoints/$a/weights_epoch_50.pt ] || SAFEONLY=1 NADAPT=500 BNTAG=_safe500 CUDA_VISIBLE_DEVICES=$g $PY bn_adapt_save.py $w purple 2>&1 | grep -E "saved|Error"
  for K in 15 20 25; do
    for h in lin gp; do
      GOALK=$K ./soft_cem.sh $g $w $M/$w/margin_${h}_fail_red_s0.pt ${t}_${h}_gr 1000 purple &
      GOALK=$K ./soft_cem.sh $g $a $M/$w/margin_${h}_fail_red_s0.pt ${t}bnsafe_${h}_gr 1000 purple &
    done
    wait
  done
}
lane 0 sigreg_only_dubins sigreg &
lane 1 sigreg_only_dubins_s1 sigreg_s1 &
lane 1 sigreg_only_dubins_s2 sigreg_s2 &
wait; echo BN_SIGREG_DONE
