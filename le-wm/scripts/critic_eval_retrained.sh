#!/usr/bin/env bash
# Eval the RETRAINED reachability critics: each critic was trained on its own
# appearance, so we score its V-sign / AUC vs HJ V* on THAT SAME appearance
# (the retrained column). Waits for both training batches to finish.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
VDIR=/data/seongbin/dreamer/lewm/_value_ood; MDIR=$VDIR/margins; LOG=/home/seongbin/latent/run_logs
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
OUT="$LOG/critic_retrained_eval.log"; : > "$OUT"

echo "[$(date +%H:%M:%S)] waiting for JEPA + dreamer retrained batches..."
until grep -q "CRITIC_RETRAINED_DONE" "$LOG/critic_retrained_batch.log" 2>/dev/null \
   && grep -q "CRITIC_RETRAINED_DREAMER_DONE" "$LOG/critic_retrained_dreamer_batch.log" 2>/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] both batches done; evaluating retrained critics." | tee -a "$OUT"

pol() { ls -d "$1"/PyHJ/gp/epoch_id_* 2>/dev/null | sort -t_ -k3 -n | tail -1; }

for tag in baseline jacobian jacpull; do
  w=${WM[$tag]}
  for cond in purple diamond rot90; do
    p=$(pol "$VDIR/jepa_${tag}_${cond}")/policy.pth
    [ -f "$p" ] || { echo "no policy jepa/$tag/$cond" | tee -a "$OUT"; continue; }
    $PY scripts/critic_ood_eval.py --tag "${tag}_${cond}" \
      --lewm_ckpt /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
      --margin_ckpt "$MDIR/${tag}_${cond}.pt" --policy "$p" 2>/dev/null \
      | grep "KEY critic .*cond=${cond} " | tee -a "$OUT"
  done
done
for cond in purple diamond rot90; do
  p=$(pol "$VDIR/dreamer_${cond}")/policy.pth
  [ -f "$p" ] || { echo "no policy dreamer/$cond" | tee -a "$OUT"; continue; }
  $PY scripts/critic_ood_eval_dreamer.py --tag "dreamer_${cond}" \
    --rssm_ckpt "$RSSM" --policy "$p" 2>/dev/null \
    | grep "KEY critic .*cond=${cond} " | tee -a "$OUT"
done
echo CRITIC_RETRAINED_EVAL_DONE | tee -a "$OUT"
