#!/usr/bin/env bash
# The 6 trained Dubins WMs of post-oct-21.md x 4 conditions, soft-cost CEM at one setting.
# Usage: [CONST=c] soft_cem_grid.sh <w>     3 models per GPU in parallel, conditions serial within a model.
cd "$(dirname "$0")"
H=/data/seongbin/dreamer/lewm; W=$1
run () { ./soft_cem.sh "$1" "$2" "$H/$2/margin_hj_color_red_rk0_s0.pt" "$3" "$W" red purple diamond fullrot90; }
run 0 sigreg_only_dubins sigreg &
run 0 jacobian_w1_dubins jacobian &
run 0 lewm_dubins_jacpull50 jacpull &
run 1 lewm_dubins_safeadv_linf50 linf &
run 1 lewm_dubins_jacpull_safeadv_linf50 jacpull_linf &
run 1 lewm_dubins_lam_bdet50 lam_bdet &
wait; echo GRID_DONE
