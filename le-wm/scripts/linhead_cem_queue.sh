#!/usr/bin/env bash
# CEM screen for adv_h with a linear internal head: safety-first soft cost (w = 1000, C = 1e4), goal cost at every imagined
# step, replan every 5 and open loop, red + purple, NEVAL episodes (default 50), with the linear AND the GP margin function.
# Each spec is <gpu>:<wm_dir>:<tag>:<heads> (heads = lin, gp or lin,gp). A wm_dir ending in _e<E> is the symlink view of
# one epoch made by epoch_head_check.sh, which also trains the margin functions there.
# Usage: linhead_cem_queue.sh <spec> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=mean NEVAL=${NEVAL:-50} CONST=10000
M=/data/seongbin/dreamer/lewm
for spec in "$@"; do
  IFS=: read -r g ck t heads <<< "$spec"
  { for hd in ${heads//,/ }; do
      for rec in 5 25; do REC=$rec ./soft_cem.sh $g $ck $M/$ck/margin_${hd}_fail_red_s0.pt ${t}_${hd}_gr 1000 red purple; done
    done; } &
done
wait; echo LINHEAD_CEM_DONE
