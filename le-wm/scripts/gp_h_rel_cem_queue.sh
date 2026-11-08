#!/usr/bin/env bash
# Arm (b) of lin_h_rank_cem_queue.sh (relative-to-current-frame cost, first 5 steps, C = 0) with the GP (MLP) margin
# function, for the linear-vs-MLP comparison. Usage: gp_h_rel_cem_queue.sh <gpu>:<wm_dir>:<tag> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=mean NEVAL=${NEVAL:-50} REC=5 HSTEPS=5 HREL=true CONST=0
H=/data/seongbin/dreamer/lewm
for spec in "$@"; do
  IFS=: read -r g ck t <<< "$spec"
  ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 red purple &
done
wait; echo GP_REL_CEM_DONE
