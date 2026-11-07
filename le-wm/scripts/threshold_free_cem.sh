#!/usr/bin/env bash
# Threshold-free elite rules (safe_solver ccem_fallback): safe_task = 50 safest by min_k h, then the 30 best goal cost;
# task_safe = 50 best goal cost, then the 30 safest. Model: jac+pull+adv_h (purple AUC ~0.99 with a linear h, sign acc
# ~0.57), linear margin function, its 3 world-model seeds. Goal distances 10-25 (crossing goals), red / purple / rot 90,
# 50 episodes. Two lanes, both on GPU0 (GPU1 is full): seeds 3072 then 2, and seed 1.
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
export NEVAL=50 GOALCOST=final CROSS=1 CCEM=1
lane () {
  local g=$1; shift
  for spec in "$@"; do
    IFS=: read -r ck t <<< "$spec"
    for k in 10 15 20 25; do
      for rule in safe_task task_safe; do
        GOALK=$k CCEMRULE=$rule ./soft_cem.sh $g $ck $M/$ck/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red purple fullrot90
      done
    done
  done
}
lane 0 lewm_dubins_jacpull_safeadv_linf50:jacpull_linf lewm_dubins_jacpull_safeadv_linf50_s2:jacpull_linf_s2 &
lane 0 lewm_dubins_jacpull_safeadv_linf50_s1:jacpull_linf_s1 &
wait; echo THRESHOLD_FREE_DONE
