#!/usr/bin/env bash
# How hard is Dubins as a function of goal distance? Goal K steps ahead (K in KS), plan horizon K executed open loop then
# replanned, episode budget 2K, final-step goal cost, 50 episodes; raw and safety-first (MLP margin function); red,
# purple, rotated 90°. K = 25 is the paper-grid protocol (paper_grid_cem.sh). Each spec <gpu>:<wm_dir>:<tag>; one lane
# per spec. Usage: [KS="5 10 15 20"] goal_dist_sweep.sh <spec> [...]
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
export NEVAL=50 GOALCOST=final
for spec in "$@"; do
  IFS=: read -r g ck t <<< "$spec"
  { for k in ${KS:-5 10 15 20}; do
      GOALK=$k CONST=0 ./soft_cem.sh $g $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gr 0 red purple fullrot90
      GOALK=$k CONST=10000 ./soft_cem.sh $g $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 red purple fullrot90
    done; } &
done
wait; echo GOAL_DIST_SWEEP_DONE
