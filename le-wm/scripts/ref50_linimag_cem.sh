#!/usr/bin/env bash
# 50-epoch reference models under the gap-reg screen protocol (gapreg_cem.sh): LINEAR h trained on uniform + predicted
# latents, CCEM minh, K=15 crossing goals, 200 episodes, red / purple / rot 90 in parallel; plus the imagined-purple
# h-ranking screen (diag_imag_color_dubins.py). Saved h files are reused. Usage: ref50_linimag_cem.sh <gpu> wm:tag ...
set -uo pipefail
cd "$(dirname "$0")"
G=$1; shift
PY=/home/seongbin/latent/le-wm/.venv/bin/python; M=/data/seongbin/dreamer/lewm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=200 GOALK=15 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
for wt in "$@"; do
  w=${wt%%:*} t=${wt#*:}
  (
    [ -f $M/$w/margin_lin_fail_red_s0.pt ] || CUDA_VISIBLE_DEVICES=$G $PY train_lin_margin_dubins.py $w 0 2>&1 | grep -E "saved|Error"
    [ -f $M/$w/margin_lin_imag_red_s0.pt ] || HEAD=lin CUDA_VISIBLE_DEVICES=$G $PY train_gp_margin_imag_dubins.py $w 0 2>&1 | grep -E "saved|Error"
    CUDA_VISIBLE_DEVICES=$G $PY diag_imag_color_dubins.py $w $M/$w/margin_gp_fail_red_s0.pt $M/$w/margin_lin_fail_red_s0.pt 2>&1 | grep -E "IMAGCOLOR|Error" &
    for c in red purple fullrot90; do ./soft_cem.sh $G $w $M/$w/margin_lin_imag_red_s0.pt ${t}_linimag_gr 1000 $c & done
    wait
  ) &
done
wait; echo REF50_DONE
