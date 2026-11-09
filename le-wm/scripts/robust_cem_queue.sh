#!/usr/bin/env bash
# Goal: models robust on real OOD samples (adv_h on the jac base) should also plan well. Same traced fair-replanning
# setting as fix_optimism_queue.sh, red + purple, on jac (reference), jac+adv_h encoder-only, jac+pull+adv_h full and
# encoder-only; h = old GP (missing baselines only), GP trained on predicted latents (imag), imag + ramp margin.
# First checks that the imag h keeps the real-sample color robustness (eval_head_color_dubins.py).
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TRACE=/data/seongbin/lewm/safeadv_results/collision_trace GOALCOST=mean REC=5 CONST=10000
PY=/home/seongbin/latent/le-wm/.venv/bin/python; H=/data/seongbin/dreamer/lewm
declare -A WM=([jacobian]=jacobian_w1_dubins [jac_linf_enc]=lewm_dubins_jac_safeadv_linf_enc50
  [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [jacpull_linf_enc]=lewm_dubins_jacpull_safeadv_linf_enc50)
for t in "${!WM[@]}"; do
  ck=${WM[$t]}
  [ -f $H/$ck/margin_gp_imag_red_s0.pt ] || CUDA_VISIBLE_DEVICES=0 $PY train_gp_margin_imag_dubins.py $ck 0 2>&1 | grep -E "unsafe frac|h \(|saved|Error"
  [ -f $H/$ck/margin_gp_fail_red_s0.pt ] || CUDA_VISIBLE_DEVICES=0 $PY train_gp_margin_dubins.py $ck 0 2>&1 | grep -E "saved|Error"
  CUDA_VISIBLE_DEVICES=0 $PY eval_head_color_dubins.py $ck $H/$ck/margin_gp_fail_red_s0.pt $H/$ck/margin_gp_imag_red_s0.pt 2>&1 | grep -E "HEADOOD|Error"
done
RAMP="[0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52,0.56,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60]"
old () { ./soft_cem.sh $1 ${WM[$2]} $H/${WM[$2]}/margin_gp_fail_red_s0.pt ${2}_gp_gr_trace 1000 $3; }
imag () { ./soft_cem.sh $1 ${WM[$2]} $H/${WM[$2]}/margin_gp_imag_red_s0.pt ${2}_gpimag_gr_trace 1000 $3; }
iramp () { MARGIN=$RAMP MTAG=ramp ./soft_cem.sh $1 ${WM[$2]} $H/${WM[$2]}/margin_gp_imag_red_s0.pt ${2}_gpimag_gr_trace 1000 $3; }
{ old 0 jacpull_linf_enc red; imag 0 jacpull_linf_enc red; iramp 0 jacpull_linf_enc red; iramp 0 jac_linf_enc red; iramp 0 jacpull_linf red; iramp 0 jacobian red; } &
{ old 0 jacpull_linf_enc purple; imag 0 jacpull_linf_enc purple; iramp 0 jacpull_linf_enc purple; imag 0 jac_linf_enc purple; iramp 0 jac_linf_enc purple; } &
{ old 1 jac_linf_enc purple; imag 1 jacpull_linf purple; iramp 1 jacpull_linf purple; imag 1 jacobian purple; iramp 1 jacobian purple; } &
wait
for p in $TRACE/*_plan_h.pt; do case $p in *jacobian*|*jac_linf_enc*|*jacpull_linf*) $PY diag_collision_cause.py "${p%_plan_h.pt}";; esac; done
echo ROBUST_DONE
