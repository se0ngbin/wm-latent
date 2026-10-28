#!/usr/bin/env bash
# GP-margin variant of the safety-first soft-cost CEM: h = the GP margin function on failure labels (same as the OOD
# tables), trained on correctly oriented red frames. Trains + saves the 6 margin functions, then 2 lanes of CEM runs
# (tag suffix _gp). Same w = 1000, C = 1e4 as the HJ-value-function grid.
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/home/seongbin/latent/le-wm/.venv/bin/python; H=/data/seongbin/dreamer/lewm
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50
               [linf]=lewm_dubins_safeadv_linf50 [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [lam_bdet]=lewm_dubins_lam_bdet50)
i=0
for t in "${!WM[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) $PY train_gp_margin_dubins.py "${WM[$t]}" 0 2>&1 | grep -E "saved|Error|Traceback" &
  i=$((i + 1))
done; wait
gp () { CONST=10000 ./soft_cem.sh "$1" "${WM[$2]}" "$H/${WM[$2]}/margin_gp_fail_red_s0.pt" "${2}_gp" 1000 "${@:3}"; }
{ for t in jacobian sigreg lam_bdet; do gp 0 $t red purple; done; for t in jacobian sigreg lam_bdet; do gp 0 $t diamond fullrot90; done; } &
{ for t in linf jacpull jacpull_linf; do gp 1 $t red purple; done; for t in linf jacpull jacpull_linf; do gp 1 $t diamond fullrot90; done; } &
wait; echo GP_QUEUE_DONE
