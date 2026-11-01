#!/usr/bin/env bash
# Fairer replanning test: goal cost = goal distance averaged over every imagined step (GOALCOST=mean) instead of the
# last step only, so receding-horizon replanning no longer pushes arrival later. Both replan intervals (REC=25 open loop,
# REC=5) under the same cost; raw CEM and GP safety-first; red; full 9-model Dubins grid.
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=mean
H=/data/seongbin/dreamer/lewm
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
  [linf]=lewm_dubins_safeadv_linf50 [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [lam_bdet]=lewm_dubins_lam_bdet50
  [jac_linf]=lewm_dubins_jac_safeadv_linf50 [jac_lam_bdet]=lewm_dubins_jac_lam_bdet50 [jacpull_lam_bdet]=lewm_dubins_jacpull_lam_bdet50)
one () {  # gpu tag: 4 runs (REC 25/5 x raw/GP safety-first)
  local g=$1 t=$2 ck=${WM[$2]} gp=$H/${WM[$2]}/margin_gp_fail_red_s0.pt
  for rec in 25 5; do
    REC=$rec ./soft_cem.sh $g $ck $gp ${t}_gr 0 red
    REC=$rec CONST=10000 ./soft_cem.sh $g $ck $gp ${t}_gp_gr 1000 red
  done
}
{ one 0 jacpull; one 0 jacpull_linf; } &
{ one 1 jacobian; one 1 jac_linf; } &
{ one 0 sigreg; one 0 linf; one 0 lam_bdet; } &
{ one 1 jac_lam_bdet; one 1 jacpull_lam_bdet; } &
wait; echo GOALMEAN_DONE
