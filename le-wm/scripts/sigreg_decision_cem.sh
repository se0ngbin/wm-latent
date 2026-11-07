#!/usr/bin/env bash
# Headline baseline (2026-11-07): sigreg (LeWM default, 1a) under the same CCEM protocol as the jac decision runs
# (minh rule, MLP h trained on uniform states, goals 15 steps out crossing an obstacle, 200 episodes, fixed eval seed
# -> paired with head_decision_cem.sh). 3 seeds in parallel.
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm
export NEVAL=200 GOALK=15 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
run () { ./soft_cem.sh $1 $2 $M/$2/margin_gp_fail_red_s0.pt ${3}_gp_gr 1000 red purple fullrot90; }
run 0 sigreg_only_dubins sigreg &
run 0 sigreg_only_dubins_s1 sigreg_s1 &
run 1 sigreg_only_dubins_s2 sigreg_s2 &
wait; echo SIGREG_DECISION_DONE
