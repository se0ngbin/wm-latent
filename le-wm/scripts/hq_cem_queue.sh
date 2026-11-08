#!/usr/bin/env bash
# Rank-normalized h (solver.h_quantile = 0.05) in the traced fair-replanning soft CEM, red + purple. Under purple the
# robust models' h keeps its ranking (AUC 0.90-0.96) but collapses its scale (balanced acc 0.500), so the sign-based
# cost never fires; ranking within each replan's candidate set is invariant to that. h = GP trained on predicted latents.
set -uo pipefail
cd "$(dirname "$0")"
export TRACE=/data/seongbin/lewm/safeadv_results/collision_trace GOALCOST=mean REC=5 CONST=10000 HQ=0.05
H=/data/seongbin/dreamer/lewm
declare -A WM=([jacobian]=jacobian_w1_dubins [jac_linf_enc]=lewm_dubins_jac_safeadv_linf_enc50
  [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [jacpull_linf_enc]=lewm_dubins_jacpull_safeadv_linf_enc50)
hq () { ./soft_cem.sh $1 ${WM[$2]} $H/${WM[$2]}/margin_gp_imag_red_s0.pt ${2}_gpimag_gr_trace 1000 red purple; }
{ hq 0 jacpull_linf_enc; hq 0 jac_linf_enc; } &
{ hq 1 jacpull_linf; hq 1 jacobian; } &
wait; echo HQ_DONE
