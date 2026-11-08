#!/usr/bin/env bash
# Decision runs (2026-11-07): linear vs MLP margin function, and linear vs MLP internal adv_h head.
# CCEM (minh rule), goals 15 steps ahead whose straight line crosses an obstacle, 200 episodes per cell (more than the
# 50 of the sweeps), red / purple / rot 90, margin functions trained on uniform states (MLP = margin_gp_fail_red_s0.pt,
# linear = margin_lin_fail_red_s0.pt). Same 200 episodes for every run (fixed eval seed) -> paired tests.
#   jac, seeds 3072 / 1 / 100: MLP h vs linear h
#   3f (jac+pull+adv_h enc, MLP internal head) vs 3g (same, linear internal head), seed 3072, each with both h
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
export NEVAL=200 GOALK=15 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
lane () {   # gpu ckpt tag heads...
  local g=$1 ck=$2 t=$3; shift 3
  for h in "$@"; do ./soft_cem.sh $g $ck $M/$ck/margin_${h}_fail_red_s0.pt ${t}_${h}_gr 1000 red purple fullrot90; done
}
lane 0 jacobian_w1_dubins jacobian lin gp &
lane 0 jacobian_w1_dubins_s1 jacobian_s1 lin gp &
lane 1 lewm_dubins_jacw1_s100 jacobian_s100 lin gp &
lane 1 lewm_dubins_jacpull_safeadv_linf_enc50 jacpull_linf_enc lin gp &
lane 1 lewm_dubins_jacpull_safeadv_linf_enc_linhead50_e50 jacpull_linf_enc_linhead lin gp &
wait; echo HEAD_DECISION_DONE
