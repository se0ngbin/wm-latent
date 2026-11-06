#!/usr/bin/env bash
# How hard is Dubins as a function of goal distance, on dangerous episodes? Goal K steps ahead, only start/goal pairs whose
# straight line crosses an obstacle (CROSS), plan horizon K executed open loop then replanned, budget 2K, final-step goal
# cost (LeWM CEM: 300 samples, 30 elites, 30 iterations), 50 episodes; per world model: raw CEM, and CCEM (SafeDreamer's
# constrained elite rule) with the MLP margin function and with the linear one; red, purple, rotated 90°.
# Phases run in order with a barrier between them (default: K 10 and 15 for every model, then 20 and 25). Each lane is
# <gpu>|<wm_dir>:<tag>[,<wm_dir>:<tag>...]; within a phase a lane goes K-major over its models. Finished cells are
# skipped (soft_cem.sh). Usage: [PHASES="10 15;20 25"] goal_dist_sweep.sh '<lane>' [...]
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
export NEVAL=50 GOALCOST=final CROSS=1
IFS=';' read -ra PH <<< "${PHASES:-10 15;20 25}"
for phase in "${PH[@]}"; do
  for lane in "$@"; do
    g=${lane%%|*}; specs=${lane#*|}
    { for k in $phase; do
        for spec in ${specs//,/ }; do
          IFS=: read -r ck t <<< "$spec"
          GOALK=$k CONST=0 ./soft_cem.sh $g $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gr 0 red purple fullrot90
          GOALK=$k CCEM=1 ./soft_cem.sh $g $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 red purple fullrot90
          GOALK=$k CCEM=1 ./soft_cem.sh $g $ck $M/$ck/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red purple fullrot90
        done
      done; } &
  done
  wait; echo "$(date +%H:%M) phase K=[$phase] done"
done
echo GOAL_DIST_SWEEP_DONE
