#!/usr/bin/env bash
# Full eval pipeline for a newly trained Dubins WM (same protocol as the 6 grid models):
#  1. value function (HJ regression, rk0) + GP margin function (failure labels), red, training orientation
#  2. fixed-orientation margin tables: color/rotate/shape x seeds 0-2 (dub_fixed_margin.py)
#  3. rollout / goal-geometry diagnostic (diag_rollout_safety.py)
#  4. CEM, 4 conditions (purple/diamond goals re-rendered by eval.py): raw, safety-first value fn, safety-first GP
# Usage: pipeline_new_wm.sh <wm_dir> <tag> <display_name> [gpu]
set -uo pipefail
cd "$(dirname "$0")"
CK=$1 TAG=$2 NAME=$3 G=${4:-1}
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/home/seongbin/latent/le-wm/.venv/bin/python; LCBF=/home/seongbin/latent/latent_cbf; MDIR=/data/seongbin/dreamer/lewm
LOG=/home/seongbin/latent/run_logs; FO=/data/seongbin/lewm/safeadv_results/fixed_orient
mkdir -p "$MDIR/$CK"
[ -f "$MDIR/$CK/margin_hj_color_red_rk0_s0.pt" ] || CUDA_VISIBLE_DEVICES=$G "$LCBF/.venv/bin/python" -u \
  "$LCBF/src/latent_cbf/scripts/train_margin_hj_lewm.py" --lewm_ckpt "$CK/weights_epoch_50.pt" \
  --out_path "$MDIR/$CK/margin_hj_color_red_rk0_s0.pt" --obstacle_color red --n_states 12000 --rollout_k 0 --steps 4000 \
  --seed 0 > "$LOG/ood_margin_train_color_${CK}_red_rk0_s0.log" 2>&1
[ -f "$MDIR/$CK/margin_gp_fail_red_s0.pt" ] || CUDA_VISIBLE_DEVICES=$G $PY train_gp_margin_dubins.py "$CK" 0 2>&1 | grep -E "saved|Error"
echo "$(date +%H:%M) $TAG margin functions ready"
for s in 0 1 2; do for ax in color rotate shape; do
  f="$FO/${TAG}_${ax}_s$s.log"; grep -q "^KEY" "$f" 2>/dev/null && continue
  DUB_ENC="$NAME=$CK" CUDA_VISIBLE_DEVICES=$G $PY dub_fixed_margin.py $ax $s > "$f" 2>&1
done; done
echo "$(date +%H:%M) $TAG margin tables: $(grep -h '^KEY' $FO/${TAG}_*.log | wc -l)/9"
CUDA_VISIBLE_DEVICES=$G $PY diag_rollout_safety.py "$TAG=$CK" 2>&1 | grep -E "^DIAG|Error"
for c in red purple diamond fullrot90; do
  ./soft_cem.sh $G $CK $MDIR/$CK/margin_gp_fail_red_s0.pt ${TAG}_gr 0 $c
  CONST=10000 ./soft_cem.sh $G $CK $MDIR/$CK/margin_hj_color_red_rk0_s0.pt ${TAG}_gr 1000 $c
  CONST=10000 ./soft_cem.sh $G $CK $MDIR/$CK/margin_gp_fail_red_s0.pt ${TAG}_gp_gr 1000 $c
done
echo "$(date +%H:%M) PIPELINE_DONE $TAG"
