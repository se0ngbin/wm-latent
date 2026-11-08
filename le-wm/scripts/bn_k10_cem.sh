#!/usr/bin/env bash
# Goal distance 10 (the one plan length where the BN-adapted MLP h's veto curve on purple has a slope): CCEM minh, crossing
# goals, 100 paired episodes, purple, jac unadapted vs BN-adapted on 500 safe purple frames, linear + MLP h, threshold 0.
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm/jacobian_w1_dubins
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh GOALK=10
for h in lin gp; do
  ./soft_cem.sh 0 jacobian_w1_dubins $M/margin_${h}_fail_red_s0.pt jacobian_${h}_gr 1000 purple &
  ./soft_cem.sh 1 jacobian_w1_dubins_bnpurple_safe500 $M/margin_${h}_fail_red_s0.pt jacobianbnsafe_${h}_gr 1000 purple &
done
wait; echo BN_K10_DONE
