#!/usr/bin/env bash
# OOD CBF eval for the margin-head critics: waits for the red driver (CBF_HEADS_ALL_DONE), then reruns only the CBF eval
# with purple obstacles (color) and diamond obstacles (shape) via cbf_margin_heads.sh COND=. Critics are reused (trained on red).
D=/home/seongbin/latent/run_logs/lewm_cbf_heads; S="$(dirname "${BASH_SOURCE[0]}")/cbf_margin_heads.sh"
J=(); for w in sigreg_only_dubins jacobian_w1_dubins lewm_dubins_jacpull50 lewm_dubins_jacpull_safeadv_linf50; do
  for t in gpfail gpimag lin; do J+=("$w:$t"); done; done
until grep -q CBF_HEADS_ALL_DONE "$D/driver.log"; do sleep 120; done
echo "[$(date +%H:%M)] red done; launching OOD"
COND=purple bash "$S" "${J[@]}" & sleep 60
COND=diamond bash "$S" "${J[@]}"; wait
echo OOD_ALL_DONE
