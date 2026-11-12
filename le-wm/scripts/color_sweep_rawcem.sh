#!/usr/bin/env bash
# Raw CEM (w = 0, no margin cost; the head arg is unused) at sweep colours alpha 0.3 / 0.4, K = 10, crossing goals,
# 100 episodes: the reference for color_sweep_ccem.sh. 7 WMs, all lanes on GPU 0 (GPU 1 is for training).
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
for spec in sigreg_only_dubins:sigreg sigreg_only_dubins_s1:sigreg_s1 sigreg_only_dubins_s2:sigreg_s2 \
            jacobian_w1_dubins:jacobian jacobian_w1_dubins_s1:jacobian_s1 lewm_dubins_jacw1_s100:jacobian_s100 \
            lewm_dubins_jac_safeadv_linf50:jac_linf lewm_dubins_jac_safeadv_linf50_s1:jac_linf_s1 lewm_dubins_jac_safeadv_linf50_s2:jac_linf_s2; do
  IFS=: read -r ck t <<< "$spec"; [ -f /data/seongbin/lewm/checkpoints/$ck/weights_epoch_50.pt ] || continue
  NEVAL=100 GOALCOST=final CROSS=1 GOALK=${K:-10} ./soft_cem.sh 0 $ck $M/$ck/margin_gp_imag_red_s0.pt ${t}_gr 0 a03 a04 &
  sleep 20
done
wait; echo COLOR_SWEEP_RAWCEM_DONE
