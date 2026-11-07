#!/usr/bin/env bash
# CCEM (minh rule) with the LINEAR margin function trained on real + PREDICTED latents (train_gp_margin_imag_dubins.py
# HEAD=lin -> margin_lin_imag_red_s0.pt), crossing goals K 10-25, red / purple / rot 90, 50 episodes.
# Models: adv_h with a linear internal head, encoder-only (only 3g is trained; 1g, 2g are not) + jac as reference.
# One lane per model on GPU G. Logs tagged <tag>_limag_gr. Usage: G=<gpu> imag_lin_ccem.sh <wm_dir>:<tag> [...]
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
for spec in "$@"; do
  IFS=: read -r ck t <<< "$spec"
  { [ -f $M/$ck/margin_lin_imag_red_s0.pt ] || HEAD=lin CUDA_VISIBLE_DEVICES=${G:-0} $PY train_gp_margin_imag_dubins.py $ck 0 2>&1 | grep -E "unsafe frac|old h|new h|saved|Error"
    for k in 10 15 20 25; do
      NEVAL=50 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh GOALK=$k ./soft_cem.sh ${G:-0} $ck $M/$ck/margin_lin_imag_red_s0.pt ${t}_limag_gr 1000 red purple fullrot90
    done; } &
done
wait; echo IMAG_LIN_CCEM_DONE
