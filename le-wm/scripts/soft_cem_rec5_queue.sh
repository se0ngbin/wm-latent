#!/usr/bin/env bash
# Is the open-loop protocol too harsh? Same sg25clean goals / 25-step horizon, but replan every 5 steps (REC=5): raw CEM
# and safety-first with the GP margin function, red and purple (purple goal re-rendered), full 9-model Dubins grid.
# Runs for models still training wait for their checkpoint + saved GP margin function (made by pipeline_new_wm.sh).
set -uo pipefail
cd "$(dirname "$0")"
export REC=5
# wait for the last two grid trainings to free GPU memory (4 CEM lanes on top of them ran out of memory)
until [ -f /data/seongbin/lewm/checkpoints/lewm_dubins_jac_safeadv_linf50/weights_epoch_50.pt ] &&
      [ -f /data/seongbin/lewm/checkpoints/lewm_dubins_jacpull_lam_bdet50/weights_epoch_50.pt ]; do sleep 300; done
H=/data/seongbin/dreamer/lewm; CK=/data/seongbin/lewm/checkpoints
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
  [linf]=lewm_dubins_safeadv_linf50 [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [lam_bdet]=lewm_dubins_lam_bdet50
  [jac_linf]=lewm_dubins_jac_safeadv_linf50 [jac_lam_bdet]=lewm_dubins_jac_lam_bdet50 [jacpull_lam_bdet]=lewm_dubins_jacpull_lam_bdet50)
one () {  # gpu tag
  local g=$1 t=$2 ck=${WM[$2]}
  until [ -f "$CK/$ck/weights_epoch_50.pt" ] && [ -f "$H/$ck/margin_gp_fail_red_s0.pt" ]; do sleep 600; done
  for c in red purple; do
    ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gr 0 $c
    CONST=10000 ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 $c
  done
}
{ one 0 jacpull; one 0 jacobian; } &
{ one 1 jacpull_linf; one 1 sigreg; } &
{ one 0 linf; one 0 lam_bdet; one 0 jac_linf; } &
{ one 1 jac_lam_bdet; one 1 jacpull_lam_bdet; } &
wait; echo REC5_DONE
