#!/usr/bin/env bash
# LeWM-style multi-seed CEM: 50 episodes per WM seed (averaged over 3 training seeds afterwards). Safety-first soft cost
# with the GP margin function (w = 1000, C = 1e4), goal cost at every imagined step, replan every 5 and open loop,
# red + purple. Each spec is <wm_dir>:<tag>; a lane runs its specs in order, waiting for the WM's GP margin function
# (seed_head_eval.sh trains it). Usage: GPU=<id> seed_cem_queue.sh <spec> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=mean NEVAL=${NEVAL:-50} CONST=10000
M=/data/seongbin/dreamer/lewm; G=${GPU:-1}
for spec in "$@"; do
  IFS=: read -r ck t <<< "$spec"
  until [ -f $M/$ck/margin_gp_fail_red_s0.pt ]; do sleep 60; done
  for rec in 5 25; do REC=$rec ./soft_cem.sh $G $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 red purple; done
done
echo SEED_CEM_DONE
