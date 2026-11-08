#!/usr/bin/env bash
# Calibrated CCEM (h(z_k) - c_k, offsets from calib_pred_dubins.py) on the only condition that passes the gate (h AUC on
# imagined latents >= 0.8 at step 15): red. jac seeds 3072 / 1 / 100, linear h on uniform states, CCEM minh, crossing goals
# at K = 10 and 15, 200 episodes; variants labeled / safe_q05, plus the uncalibrated run at K=10 (K=15 exists).
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python
OFF=/data/seongbin/lewm/dubins_calib/offsets_pred.json
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=200 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
offs () { $PY -c "import json; print(json.load(open('$OFF'))['$1']['margin_lin_fail_red_s0.pt']['red']['$2'])"; }
i=0
for wt in jacobian_w1_dubins:jacobian jacobian_w1_dubins_s1:jacobian_s1 lewm_dubins_jacw1_s100:jacobian_s100; do
  w=${wt%%:*} t=${wt#*:}
  for K in 10 15; do
    g=$((i % 2)); i=$((i + 1))
    ( export GOALK=$K
      [ $K = 10 ] && ./soft_cem.sh $g $w $M/$w/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red &
      for v in labeled safe_q05; do
        HOFFS="$(offs $w $v)" CTAG=$v ./soft_cem.sh $g $w $M/$w/margin_lin_fail_red_s0.pt ${t}_lin_gr 1000 red &
      done
      wait ) &
  done
done
wait; echo CALIB_CEM_RED_DONE
