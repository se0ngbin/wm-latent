#!/usr/bin/env bash
# (1) h control: safety-first CEM on jac (red) with margin functions that know nothing about safety (random init,
#     shuffled labels) -- is the collision drop from h's safety knowledge?
# (2) purple / diamond reruns with the goal frame re-rendered in the shifted appearance (eval.py patch), tag suffix _gr:
#     raw CEM, safety-first with the supervised value function, safety-first with the GP margin function, 6 models.
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/home/seongbin/latent/le-wm/.venv/bin/python; H=/data/seongbin/dreamer/lewm
CUDA_VISIBLE_DEVICES=1 $PY make_control_heads.py jacobian_w1_dubins 2>&1 | grep -E "saved|Error|Traceback"
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
               [linf]=lewm_dubins_safeadv_linf50 [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [lam_bdet]=lewm_dubins_lam_bdet50)
J=jacobian_w1_dubins
lane () {  # gpu model-tags...
  local g=$1; shift
  for t in "$@"; do ck=${WM[$t]}
    for c in purple diamond; do
      ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gr 0 $c
      CONST=10000 ./soft_cem.sh $g $ck $H/$ck/margin_hj_color_red_rk0_s0.pt ${t}_gr 1000 $c
      CONST=10000 ./soft_cem.sh $g $ck $H/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 $c
    done
  done
}
{ CONST=10000 ./soft_cem.sh 1 $J $H/$J/margin_ctrl_random.pt jacobian_ctrlrand 1000 red; lane 1 sigreg lam_bdet; } &
{ CONST=10000 ./soft_cem.sh 0 $J $H/$J/margin_ctrl_shuffled.pt jacobian_ctrlshuf 1000 red; lane 0 jacobian linf; } &
{ lane 1 jacpull jacpull_linf; } &
wait; echo GOALFIX_DONE
