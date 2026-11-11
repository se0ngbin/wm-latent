#!/usr/bin/env bash
# Red -> purple obstacle colour sweep on the Dubins 3x3 grid (3 WM seeds; untrained WMs are skipped = blank cells).
# Per WM: train the missing real+imag margin functions (GP and linear, train_gp_margin_imag_dubins.py, seed 0; skipped if
# saved), then color_sweep_dubins.py on both. LANES parallel lanes on GPU G. Logs: run_logs/color_sweep/<wm>.log
set -uo pipefail
cd "$(dirname "$0")"
M=/data/seongbin/dreamer/lewm; C=/data/seongbin/lewm/checkpoints; PY=/home/seongbin/latent/le-wm/.venv/bin/python
L=/home/seongbin/latent/run_logs/color_sweep; mkdir -p $L
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${G:-0}
WMS=()
for b in sigreg_only_dubins lewm_dubins_safeadv_linf50 lewm_dubins_lam_bdet50 lewm_dubins_jac_safeadv_linf50 \
         lewm_dubins_jac_lam_bdet50 lewm_dubins_jacpull50 lewm_dubins_jacpull_safeadv_linf50 lewm_dubins_jacpull_lam_bdet50; do
  for s in "" _s1 _s2; do [ -f $C/$b$s/weights_epoch_50.pt ] && WMS+=($b$s); done
done
WMS=(jacobian_w1_dubins jacobian_w1_dubins_s1 lewm_dubins_jacw1_s100 "${WMS[@]}")
run() {
  local ck=$1
  for hd in gp lin; do
    [ -f $M/$ck/margin_${hd}_imag_red_s0.pt ] || HEAD=$hd $PY train_gp_margin_imag_dubins.py $ck 0
  done
  $PY color_sweep_dubins.py $ck $M/$ck/margin_gp_imag_red_s0.pt $M/$ck/margin_lin_imag_red_s0.pt
}
for ck in "${WMS[@]}"; do
  while [ $(jobs -rp | wc -l) -ge ${LANES:-3} ]; do wait -n; done
  run $ck > $L/$ck.log 2>&1 &
done
wait; echo COLOR_SWEEP_DONE
