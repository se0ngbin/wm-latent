#!/usr/bin/env bash
# Run calib_pred_dubins.py (per-step threshold calibration + per-step AUC gate) over the trained Dubins models with
# their uniform-state linear and MLP margin functions (skipping missing ones), split over two GPUs.
# Output: run_logs/calib_pred_screen_<gpu>.log, offsets in /data/seongbin/lewm/dubins_calib/offsets_pred.json.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/seongbin/latent/le-wm/.venv/bin/python; M=/data/seongbin/dreamer/lewm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
lane () {
  local g=$1; shift
  for w in "$@"; do
    local hs=()
    for h in lin gp; do [ -f $M/$w/margin_${h}_fail_red_s0.pt ] && hs+=($M/$w/margin_${h}_fail_red_s0.pt); done
    [ ${#hs[@]} -gt 0 ] && CUDA_VISIBLE_DEVICES=$g $PY calib_pred_dubins.py $w "${hs[@]}" 2>&1 | grep -E "CALIB|  k=|Error"
  done > /home/seongbin/latent/run_logs/calib_pred_screen_$g.log
}
lane 0 jacobian_w1_dubins_s1 lewm_dubins_jacpull50 lewm_dubins_jacpull_safeadv_linf_enc50 lewm_dubins_jac_lam_bdet50 \
  lewm_dubins_jac_safeadv_linf50 lewm_dubins_jac_safeadv_linf_enc50 sigreg_only_dubins &
lane 1 lewm_dubins_jacw1_s100 lewm_dubins_jacpull_safeadv_linf_enc_linhead50_e50 lewm_dubins_jacpull_lam_bdet50 \
  lewm_dubins_jacpull_safeadv_linf50 lewm_dubins_jacpull_lam_bdet_enc50 lewm_dubins_jac_predgap15_e50 &
wait; echo CALIB_SCREEN_DONE
