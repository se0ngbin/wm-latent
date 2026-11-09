#!/usr/bin/env bash
# CCEM with a test-time BN-adapted WM (bn_adapt_save.py) on its condition, h = the original WM's red-trained margin
# functions. Protocol as fs5_cem.sh (minh, crossing goals, K = 15 / 20 / 25, 100 episodes) -> paired with the unadapted
# runs softcem_<tag>_<h>_gr_w1000_k<K>_cross_ccemminh_n100_<cond>.log. Usage: bn_cem.sh <gpu> <wm> <tag> <cond>
set -uo pipefail
cd "$(dirname "$0")"
G=$1 W=$2 T=$3 C=$4
M=/data/seongbin/dreamer/lewm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
for K in 15 20 25; do
  for h in lin gp; do GOALK=$K ./soft_cem.sh $G ${W}_bn$C $M/$W/margin_${h}_fail_red_s0.pt ${T}bn${C}_${h}_gr 1000 $C & done
done
wait; echo BN_CEM_DONE
