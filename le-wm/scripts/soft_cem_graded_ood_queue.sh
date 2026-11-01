#!/usr/bin/env bash
# Graded GP soft cost (hinge only, C = 0, w = 1000) on all four conditions (purple / diamond with a re-rendered goal
# frame, rotated; red already exists for the safety-reg models). Same models as the red graded sweep
# (soft_cem_weight_pair_gp.sh: 6 full + 6 encoder-only safety-reg WMs) plus sigreg / jac / jac+pull as references.
# Log tag <model>_gp_weight, matching the red sweep (finished cells are skipped). Two lanes.
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
H=/data/seongbin/dreamer/lewm
declare -A WM=(
  [sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
  [linf]=lewm_dubins_safeadv_linf50 [jac_linf]=lewm_dubins_jac_safeadv_linf50 [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50
  [lam_bdet]=lewm_dubins_lam_bdet50 [jac_lam_bdet]=lewm_dubins_jac_lam_bdet50 [jacpull_lam_bdet]=lewm_dubins_jacpull_lam_bdet50
  [sa_enc]=lewm_dubins_sa_enc50 [jac_linf_enc]=lewm_dubins_jac_safeadv_linf_enc50
  [jacpull_linf_enc]=lewm_dubins_jacpull_safeadv_linf_enc50 [lam_bdet_enc]=lewm_dubins_lam_bdet_enc50
  [jac_lam_bdet_enc]=lewm_dubins_jac_lam_bdet_enc50 [jacpull_lam_bdet_enc]=lewm_dubins_jacpull_lam_bdet_enc50)
lane () {  # gpu tags...
  local g=$1; shift
  for t in "$@"; do ck=${WM[$t]}; ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gp_weight 1000 red purple diamond fullrot90; done
}
lane 0 jacobian jac_linf jac_linf_enc jac_lam_bdet jac_lam_bdet_enc jacpull_lam_bdet_enc linf lam_bdet &
lane 1 jacpull jacpull_linf jacpull_linf_enc jacpull_lam_bdet sigreg sa_enc lam_bdet_enc &
wait; echo GRADED_OOD_DONE
