#!/usr/bin/env bash
# Red (alpha 0) and purple (alpha 1) anchors for the colour-sweep CEM tables: CCEM (real+imag GP margin function) and raw
# CEM, same protocol (crossing goals, 100 episodes, goal frames re-rendered for purple), K = 10 then K = 15, on GPU ${G:-1}.
# CCEM and raw run in parallel per K (2 x 7 lanes). Finished runs are skipped by soft_cem.sh.
set -uo pipefail
cd "$(dirname "$0")"
L=/home/seongbin/latent/run_logs
for k in 10 15; do
  G=${G:-1} K=$k SCENS="red purple" ./color_sweep_ccem.sh > $L/color_sweep_anchors_ccem_k$k.log 2>&1 &
  G=${G:-1} K=$k SCENS="red purple" ./color_sweep_rawcem.sh > $L/color_sweep_anchors_raw_k$k.log 2>&1 &
  wait; echo "$(date +%H:%M) K=$k anchors done"
done
echo COLOR_SWEEP_ANCHORS_DONE
