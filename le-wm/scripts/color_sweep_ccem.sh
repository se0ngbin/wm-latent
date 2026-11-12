#!/usr/bin/env bash
# CCEM (minh rule) at intermediate obstacle colours (red->purple sweep alpha 0.3 = a03, then 0.4 = a04), K = 10, crossing
# goals, 100 episodes, real+imag GP margin function (margin_gp_imag_red_s0.pt). 7 WMs: sigreg x3, jac x3, jac+adv_h (s3072).
# One lane per WM; lane i on GPU ${GPUS[i]}. Logs: run_logs/softcem_<tag>_gpimag_gr_w1000_k10_cross_ccemminh_n100_a0{3,4}.log
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
SPECS=(sigreg_only_dubins:sigreg sigreg_only_dubins_s1:sigreg_s1 sigreg_only_dubins_s2:sigreg_s2
       jacobian_w1_dubins:jacobian jacobian_w1_dubins_s1:jacobian_s1 lewm_dubins_jacw1_s100:jacobian_s100
       lewm_dubins_jac_safeadv_linf50:jac_linf lewm_dubins_jac_safeadv_linf50_s1:jac_linf_s1 lewm_dubins_jac_safeadv_linf50_s2:jac_linf_s2)
GPUS=(${GPUS:-$(printf "${G:-0} %.0s" 1 2 3 4 5 6 7 8 9)})
for i in "${!SPECS[@]}"; do
  IFS=: read -r ck t <<< "${SPECS[$i]}"; [ -f /data/seongbin/lewm/checkpoints/$ck/weights_epoch_50.pt ] || continue
  NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh GOALK=${K:-10} \
    ./soft_cem.sh ${GPUS[$i]} $ck $M/$ck/margin_gp_imag_red_s0.pt ${t}_gpimag_gr 1000 ${SCENS:-a03 a04} &
  sleep 20
done
wait; echo COLOR_SWEEP_CCEM_DONE
