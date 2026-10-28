#!/usr/bin/env bash
# Penalty-weight sweep, red only: jacobian on GPU0, sigreg only on GPU1, 3 weights each in parallel.
cd "$(dirname "$0")"
H=/data/seongbin/dreamer/lewm
for w in 100 1000 10000; do
  ./soft_cem.sh 0 jacobian_w1_dubins $H/jacobian_w1_dubins/margin_hj_color_red_rk0_s0.pt jacobian $w red &
  ./soft_cem.sh 1 sigreg_only_dubins $H/sigreg_only_dubins/margin_hj_color_red_rk0_s0.pt sigreg $w red &
done
wait; echo SWEEP_DONE
