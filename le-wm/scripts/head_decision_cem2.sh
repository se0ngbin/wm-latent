#!/usr/bin/env bash
# Decision runs, CCEM variants 2 and 3 (variant 1 = uniform-trained h is head_decision_cem.sh). Same models, goals
# (15 steps, crossing), 200 episodes, red / purple / rot 90:
#   variant 2: CCEM (minh rule) with h trained on real + PREDICTED latents (train_gp_margin_imag_dubins.py, both the MLP
#              and the linear h; missing ones are trained first), logs tagged <tag>_<h>imag_gr
#   variant 3: threshold-free safe_task_norm (h z-scored per imagined step across candidates, 50 safest by the worst
#              normalized step, then the 30 best goal cost) with the uniform-trained h, logs <tag>_<h>_gr ..._ccemsafe_task_norm
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=200 GOALK=15 GOALCOST=final CROSS=1 CCEM=1
lane () {   # gpu ckpt tag
  local g=$1 ck=$2 t=$3
  for h in lin gp; do
    [ -f $M/$ck/margin_${h}_imag_red_s0.pt ] || HEAD=$h CUDA_VISIBLE_DEVICES=$g $PY train_gp_margin_imag_dubins.py $ck 0 2>&1 | grep -E "old h|new h|saved|Error"
  done
  for h in lin gp; do CCEMRULE=minh ./soft_cem.sh $g $ck $M/$ck/margin_${h}_imag_red_s0.pt ${t}_${h}imag_gr 1000 red purple fullrot90; done
  for h in lin gp; do CCEMRULE=safe_task_norm ./soft_cem.sh $g $ck $M/$ck/margin_${h}_fail_red_s0.pt ${t}_${h}_gr 1000 red purple fullrot90; done
}
lane 0 jacobian_w1_dubins jacobian &
lane 1 jacobian_w1_dubins_s1 jacobian_s1 &
lane 1 lewm_dubins_jacw1_s100 jacobian_s100 &
lane 1 lewm_dubins_jacpull_safeadv_linf_enc50 jacpull_linf_enc &
lane 0 lewm_dubins_jacpull_safeadv_linf_enc_linhead50_e50 jacpull_linf_enc_linhead &
wait; echo HEAD_DECISION2_DONE
