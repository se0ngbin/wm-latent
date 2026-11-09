#!/usr/bin/env bash
# Evaluation of jac+predgap at 50 epochs (lewm_dubins_jac_predgap50) against jac 50 (jacobian_w1_dubins), same protocols:
#   1. margin functions (linear + MLP, red uniform states; saved and reused)
#   2. imagined-state screen (diag_imag_fs_dubins.py: per-step AUC red / purple / rot 90) for both models
#   3. veto-vs-distance curve (veto_curve_dubins.py) at plan lengths 10 and 15, red and purple, both models
#   4. CCEM minh, crossing goals, K = 10 / 15 / 20, 100 episodes, red / purple / rot 90, linear + MLP h (jac's missing
#      K=10 red / rot 90 references included; existing jac logs are skipped by soft_cem.sh), paired by episode with jac
# Logs: run_logs/predgap50_{screen,veto}.log and softcem_{jacpredgap50,jacobian}_*_n100_*.log
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/seongbin/latent/le-wm/.venv/bin/python; M=/data/seongbin/dreamer/lewm; L=/home/seongbin/latent/run_logs
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
P=lewm_dubins_jac_predgap50; J=jacobian_w1_dubins
[ -f $M/$P/margin_gp_fail_red_s0.pt ] || CUDA_VISIBLE_DEVICES=0 $PY train_gp_margin_dubins.py $P 0 2>&1 | grep -E "saved|Error"
[ -f $M/$P/margin_lin_fail_red_s0.pt ] || CUDA_VISIBLE_DEVICES=0 $PY train_lin_margin_dubins.py $P 0 2>&1 | grep -E "saved|Error"
(
  for w in $P $J; do
    CUDA_VISIBLE_DEVICES=1 $PY diag_imag_fs_dubins.py $w 1 25 $M/$w/margin_lin_fail_red_s0.pt $M/$w/margin_gp_fail_red_s0.pt 2>&1 | grep -E "IMAGFS|Error"
  done > $L/predgap50_screen.log
  for K in 10 15; do for c in red purple; do for w in $P $J; do
    K=$K CUDA_VISIBLE_DEVICES=1 $PY veto_curve_dubins.py $w $c $M/$w/margin_lin_fail_red_s0.pt $M/$w/margin_gp_fail_red_s0.pt 2>&1 | grep -E "VETO|Error" | sed "s/^/K$K /"
  done; done; done > $L/predgap50_veto.log
) &
export NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
for K in 10 15 20; do
  for h in lin gp; do for c in red purple fullrot90; do
    GOALK=$K ./soft_cem.sh 0 $P $M/$P/margin_${h}_fail_red_s0.pt jacpredgap50_${h}_gr 1000 $c &
    GOALK=$K ./soft_cem.sh 0 $J $M/$J/margin_${h}_fail_red_s0.pt jacobian_${h}_gr 1000 $c &
  done; done
  wait
done
wait; echo PREDGAP50_EVAL_DONE
