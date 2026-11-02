#!/usr/bin/env bash
# Two fixes for the optimistic h on imagined latents (executed-plan traces: 85-100% of collisions false-safe), same
# setting as collision_cause_queue.sh (GP safety-first, w = 1000, C = 1e4, goal_cost=mean, replan every 5, red, TRACE on):
#   ramp: per-imagined-step margin e_t = min(0.6, 0.04 (t+1)), from the traces (crash-step h grows with lookahead:
#         jac p75 0.17 at 1-5 steps ahead, 0.51 at 6-10, 0.8-1.0 beyond; safe-plan h stays flat);
#   imag: GP margin function trained on predicted latents of CEM-style plans labelled in the sim (train_gp_margin_imag_dubins.py).
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TRACE=/data/seongbin/lewm/safeadv_results/collision_trace GOALCOST=mean REC=5 CONST=10000
PY=/home/seongbin/latent/le-wm/.venv/bin/python; H=/data/seongbin/dreamer/lewm
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
  [linf]=lewm_dubins_safeadv_linf50 [jac_linf]=lewm_dubins_jac_safeadv_linf50 [jac_linf_enc]=lewm_dubins_jac_safeadv_linf_enc50
  [jac_lam_bdet]=lewm_dubins_jac_lam_bdet50 [jac_lam_bdet_enc]=lewm_dubins_jac_lam_bdet_enc50
  [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50)
for t in "${!WM[@]}"; do
  [ -f $H/${WM[$t]}/margin_gp_imag_red_s0.pt ] || CUDA_VISIBLE_DEVICES=0 $PY train_gp_margin_imag_dubins.py ${WM[$t]} 0 2>&1 | grep -E "unsafe frac|h \(|saved|Error"
done
RAMP="[0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52,0.56,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60]"
ramp () { MARGIN=$RAMP MTAG=ramp ./soft_cem.sh $1 ${WM[$2]} $H/${WM[$2]}/margin_gp_fail_red_s0.pt ${2}_gp_gr_trace 1000 red; }
imag () { ./soft_cem.sh $1 ${WM[$2]} $H/${WM[$2]}/margin_gp_imag_red_s0.pt ${2}_gpimag_gr_trace 1000 red; }
{ for t in jacobian jac_linf jac_lam_bdet jacpull_linf sigreg; do ramp 0 $t; done; } &
{ for t in jacobian jac_linf jac_lam_bdet jacpull_linf sigreg; do imag 0 $t; done; } &
{ for t in jacpull jac_linf_enc jac_lam_bdet_enc linf; do ramp 1 $t; done; } &
{ for t in jacpull jac_linf_enc jac_lam_bdet_enc linf; do imag 1 $t; done; } &
wait
for p in $TRACE/*_mramp_red_plan_h.pt $TRACE/*gpimag*_plan_h.pt; do $PY diag_collision_cause.py "${p%_plan_h.pt}"; done
echo FIX_DONE
