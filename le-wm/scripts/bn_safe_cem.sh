#!/usr/bin/env bash
# Realistic test-time adaptation (2026-11-08): BN statistics adapted on NADAPT=500 SAFE unlabeled purple frames
# (bn_adapt_save.py SAFEONLY=1 -> <wm>_bnpurple_safe500), and the CCEM threshold set from the same safe frames
# (c = 10% quantile of h over them, from bnsafe_screen_bn1_n500.log: linear +0.00, MLP +0.43; constant over steps).
# CCEM minh, crossing goals, K = 15 / 20 / 25, 100 episodes, purple; h = jac's red-trained margin functions.
# Paired with softcem_jacobian_<h>_gr_w1000_k<K>_cross_ccemminh_n100_purple.log (unadapted).
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
W=jacobian_w1_dubins; A=${W}_bnpurple_safe500
[ -f $STABLEWM_HOME/checkpoints/$A/weights_epoch_50.pt ] || SAFEONLY=1 NADAPT=500 BNTAG=_safe500 CUDA_VISIBLE_DEVICES=1 $PY bn_adapt_save.py $W purple 2>&1 | grep -E "saved|Error"
export NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
declare -A Q10=([lin]=0.00 [gp]=0.43)
i=0
for K in 15 20 25; do
  for h in lin gp; do
    g=$((i % 2)); i=$((i + 1))
    GOALK=$K ./soft_cem.sh $g $A $M/$W/margin_${h}_fail_red_s0.pt jacobianbnsafe_${h}_gr 1000 purple &
    GOALK=$K HOFFS="[${Q10[$h]}]" CTAG=safeq10 ./soft_cem.sh $g $A $M/$W/margin_${h}_fail_red_s0.pt jacobianbnsafe_${h}_gr 1000 purple &
  done
done
wait; echo BN_SAFE_CEM_DONE
