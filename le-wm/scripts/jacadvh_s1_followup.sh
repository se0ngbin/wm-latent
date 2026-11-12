#!/usr/bin/env bash
# After jac+adv_h Dubins WM seed 1 finishes: real+imag margin functions + colour sweep (color_sweep_dubins.sh), then raw
# CEM and CCEM at K = 10 and 15, alpha 0.3 / 0.4 + red / purple anchors (finished runs of other WMs are skipped by soft_cem.sh). All on GPU ${G:-0}.
# Usage: jacadvh_s1_followup.sh <training pid>
set -uo pipefail
cd "$(dirname "$0")"
L=/home/seongbin/latent/run_logs
while kill -0 "$1" 2>/dev/null; do sleep 120; done
[ -f /data/seongbin/lewm/checkpoints/lewm_dubins_jac_safeadv_linf50_s1/weights_epoch_50.pt ] \
  || { echo "$(date +%H:%M) seed 1 exited without weights_epoch_50.pt; nothing run"; exit 1; }
echo "$(date +%H:%M) seed 1 done; margin functions + sweep"
G=${G:-0} LANES=1 ./color_sweep_dubins.sh
echo "$(date +%H:%M) sweep done; CEM"
SCENS="a03 a04 red purple" G=${G:-0} K=10 ./color_sweep_ccem.sh > $L/color_sweep_ccem_s1_k10.log 2>&1 &
SCENS="a03 a04 red purple" G=${G:-0} K=15 ./color_sweep_ccem.sh > $L/color_sweep_ccem_s1_k15.log 2>&1 &
SCENS="a03 a04 red purple" G=${G:-0} K=10 ./color_sweep_rawcem.sh > $L/color_sweep_rawcem_s1_k10.log 2>&1 &
SCENS="a03 a04 red purple" G=${G:-0} K=15 ./color_sweep_rawcem.sh > $L/color_sweep_rawcem_s1_k15.log 2>&1 &
wait; echo "$(date +%H:%M) JACADVH_S1_FOLLOWUP_DONE"
