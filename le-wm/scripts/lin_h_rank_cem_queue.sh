#!/usr/bin/env bash
# Threshold-free / rank-based costs with the LINEAR margin function, which keeps the ranking on purple but not the sign.
# Both arms: safety cost on the first 5 imagined steps, replan every 5, goal cost at every imagined step, w = 1000,
# NEVAL episodes (default 50), red + purple.
#   (a) dynamic threshold: HQ=0.05, C = 1e4 (same as the GP hs5_hq0.05 runs)
#   (b) relative to the current frame: HREL, penalty relu(h(z_0) - h(z_t)), C = 0 (any decrease would count as a violation)
# Usage: lin_h_rank_cem_queue.sh <gpu>:<wm_dir>:<tag> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=mean NEVAL=${NEVAL:-50} REC=5 HSTEPS=5
H=/data/seongbin/dreamer/lewm
for spec in "$@"; do
  IFS=: read -r g ck t <<< "$spec"
  { HQ=0.05 CONST=10000 ./soft_cem.sh $g $ck $H/$ck/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red purple
    HREL=true CONST=0 ./soft_cem.sh $g $ck $H/$ck/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red purple; } &
done
wait; echo LIN_RANK_CEM_DONE
