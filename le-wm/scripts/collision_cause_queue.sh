#!/usr/bin/env bash
# Why does safety-first CEM still collide? Same setting as the best fair-replanning cell (GP safety-first, w = 1000,
# C = 1e4, goal_cost=mean, replan every 5 steps, red), rerun with TRACE on; then diag_collision_cause.py splits each
# collision into "plan predicted unsafe" vs "false-safe".
set -uo pipefail
cd "$(dirname "$0")"
export TRACE=/data/seongbin/lewm/safeadv_results/collision_trace GOALCOST=mean REC=5 CONST=10000
mkdir -p $TRACE
H=/data/seongbin/dreamer/lewm
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
  [linf]=lewm_dubins_safeadv_linf50 [jac_linf]=lewm_dubins_jac_safeadv_linf50 [jac_linf_enc]=lewm_dubins_jac_safeadv_linf_enc50
  [jac_lam_bdet]=lewm_dubins_jac_lam_bdet50 [jac_lam_bdet_enc]=lewm_dubins_jac_lam_bdet_enc50
  [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50)
lane () { local g=$1; shift; for t in "$@"; do ck=${WM[$t]}; ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr_trace 1000 red; done; }
lane 0 jacobian jac_linf jac_linf_enc jacpull_linf linf &
lane 1 jac_lam_bdet jac_lam_bdet_enc jacpull sigreg &
wait
for p in $TRACE/*_plan_h.pt; do /home/seongbin/latent/le-wm/.venv/bin/python diag_collision_cause.py "${p%_plan_h.pt}"; done
echo CAUSE_DONE
