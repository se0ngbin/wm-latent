#!/usr/bin/env bash
# Lexicographic variant (w=1000 hinge + 1e4 flat violation cost), red; waits for the w-sweep to finish.
cd "$(dirname "$0")"
while pgrep -f "soft_cem_sweep.sh" >/dev/null; do sleep 60; done
H=/data/seongbin/dreamer/lewm
CONST=10000 ./soft_cem.sh 0 jacobian_w1_dubins $H/jacobian_w1_dubins/margin_hj_color_red_rk0_s0.pt jacobian 1000 red &
CONST=10000 ./soft_cem.sh 1 sigreg_only_dubins $H/sigreg_only_dubins/margin_hj_color_red_rk0_s0.pt sigreg 1000 red &
wait; echo LEX_DONE
