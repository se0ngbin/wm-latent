#!/usr/bin/env bash
# Frameskip-5 jac+pull vs references (2026-11-08): CCEM minh, crossing goals K = 15 / 20 / 25 env steps (frameskip-5
# models plan K/5 blocks of 5), 100 episodes, red / purple / rot 90, linear + MLP h on uniform red states (trained once,
# saved, reused). Same eval seed for every run -> paired per K. Logs: run_logs/softcem_<tag>_<h>_gr_w1000_k<K>[_fs5]_cross_ccemminh_n100_<cond>.log
#   wm:tag:fs   fs5 jac+pull waits for its weights_epoch_50.pt (train_fs5_jacpull.sh)
# Usage: fs5_cem.sh <gpu> wm:tag:fs ...
set -uo pipefail
cd "$(dirname "$0")"
G=$1; shift
M=/data/seongbin/dreamer/lewm; PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=100 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
for spec in "$@"; do
  IFS=: read -r w t fs <<< "$spec"
  until [ -f $STABLEWM_HOME/checkpoints/$w/weights_epoch_50.pt ]; do sleep 300; done
  [ -f $M/$w/margin_gp_fail_red_s0.pt ] || CUDA_VISIBLE_DEVICES=$G $PY train_gp_margin_dubins.py $w 0 2>&1 | grep -E "saved|Error"
  [ -f $M/$w/margin_lin_fail_red_s0.pt ] || CUDA_VISIBLE_DEVICES=$G $PY train_lin_margin_dubins.py $w 0 2>&1 | grep -E "saved|Error"
  for K in 15 20 25; do
    for h in lin gp; do for c in red purple fullrot90; do
      FS=$fs GOALK=$K ./soft_cem.sh $G $w $M/$w/margin_${h}_fail_red_s0.pt ${t}_${h}_gr 1000 $c &
    done; done
    wait
  done
done
echo FS5_CEM_DONE
