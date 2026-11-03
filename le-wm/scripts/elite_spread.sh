#!/usr/bin/env bash
# Do CEM's elites converge? jac, red, open loop, final-step goal cost: GP safety-first vs raw (w = 0), with the
# _EliteSpread callback on (SOFTCEM_SPREAD); analysis: diag_elite_spread.py.
set -uo pipefail
cd "$(dirname "$0")"
OUT=/data/seongbin/lewm/safeadv_results/elite_spread; mkdir -p $OUT
H=/data/seongbin/dreamer/lewm/jacobian_w1_dubins
SOFTCEM_SPREAD=$OUT/jac_sf.pt CONST=10000 ./soft_cem.sh 0 jacobian_w1_dubins $H/margin_gp_fail_red_s0.pt jacobian_spread 1000 red &
SOFTCEM_SPREAD=$OUT/jac_raw.pt ./soft_cem.sh 1 jacobian_w1_dubins $H/margin_gp_fail_red_s0.pt jacobian_spread 0 red &
wait
echo SPREAD_DONE
