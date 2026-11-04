#!/usr/bin/env bash
# Linear margin function (train_lin_margin_dubins.py) as h in the safety-first soft CEM (w = 1000, C = 1e4), goal cost at
# every imagined step, replan every 5 and open loop, red + purple, NEVAL episodes (default 50). One lane per WM.
# Usage: lin_h_cem_queue.sh <gpu>:<wm_dir>:<tag> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=mean NEVAL=${NEVAL:-50} CONST=10000
H=/data/seongbin/dreamer/lewm
for spec in "$@"; do
  IFS=: read -r g ck t <<< "$spec"
  { REC=5 ./soft_cem.sh $g $ck $H/$ck/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red purple
    REC=25 ./soft_cem.sh $g $ck $H/$ck/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red purple; } &
done
wait; echo LIN_H_CEM_DONE
