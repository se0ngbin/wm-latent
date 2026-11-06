#!/usr/bin/env bash
# How hard is Dubins as a function of goal distance, on dangerous episodes? Goal K steps ahead (K in KS), only start/goal
# pairs whose straight line crosses an obstacle (CROSS), plan horizon K executed open loop then replanned, budget 2K,
# final-step goal cost (LeWM CEM: 300 samples, 30 elites, 30 iterations), 50 episodes; raw CEM and SafeDreamer's
# constrained-CEM elite rule (CCEM, MLP margin function as the cost model); red, purple, rotated 90°.
# Each spec <gpu>:<wm_dir>:<tag>; one lane per spec. Usage: [KS="10 15 20 25"] goal_dist_sweep.sh <spec> [...]
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
export NEVAL=50 GOALCOST=final CROSS=1
for spec in "$@"; do
  IFS=: read -r g ck t <<< "$spec"
  { for k in ${KS:-10 15 20 25}; do
      GOALK=$k CONST=0 ./soft_cem.sh $g $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gr 0 red purple fullrot90
      GOALK=$k CCEM=1 ./soft_cem.sh $g $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 red purple fullrot90
    done; } &
done
wait; echo GOAL_DIST_SWEEP_DONE
