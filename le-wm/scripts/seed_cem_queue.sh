#!/usr/bin/env bash
# LeWM-style multi-seed CEM: 50 episodes per WM seed (averaged over 3 training seeds afterwards). Safety-first soft cost
# with the GP margin function (w = 1000, C = 1e4), red, purple and rotated 90° (SCENS). Default: open loop (RECS=25) with the final-step goal
# cost (GOALCOST=final), the main-table protocol, so the 200-episode seed-3072 runs are the third seed. Replanning is
# RECS="5" GOALCOST=mean. Each spec is <wm_dir>:<tag>; a lane runs its specs in order, waiting for the WM's GP margin
# function (seed_head_eval.sh trains it). Usage: GPU=<id> [RECS="25"] [GOALCOST=final] seed_cem_queue.sh <spec> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=${GOALCOST:-final} NEVAL=${NEVAL:-50} CONST=10000
M=/data/seongbin/dreamer/lewm; G=${GPU:-1}
for spec in "$@"; do
  IFS=: read -r ck t <<< "$spec"
  until [ -f $M/$ck/margin_gp_fail_red_s0.pt ]; do sleep 60; done
  for rec in ${RECS:-25}; do REC=$rec ./soft_cem.sh $G $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 ${SCENS:-red purple fullrot90}; done
done
echo SEED_CEM_DONE
