#!/usr/bin/env bash
# Eval the in-dist (red) reachability critics on all appearance conditions:
# sign-acc / AUC of V_latent vs HJ V*. red = in-dist, purple/diamond/rot90 = zero-shot.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
VDIR=/data/seongbin/dreamer/lewm/_value_ood; LOG=/home/seongbin/latent/run_logs
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
for tag in baseline jacobian jacpull; do
  w=${WM[$tag]}
  pol=$(ls -d "$VDIR/jepa_$tag"/PyHJ/gp/epoch_id_* 2>/dev/null | sort -t_ -k3 -n | tail -1)/policy.pth
  [ -f "$pol" ] || { echo "no policy for $tag"; continue; }
  echo "[$(date +%H:%M:%S)] eval critic $tag  ($pol)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/critic_ood_eval.py --tag "$tag" \
    --lewm_ckpt /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
    --margin_ckpt /data/seongbin/dreamer/lewm/$w/margin_heads.pt --policy "$pol" \
    > "$LOG/critic_eval_$tag.log" 2>&1
  grep "KEY critic" "$LOG/critic_eval_$tag.log" || tail -3 "$LOG/critic_eval_$tag.log"
done
# Dreamer baseline critic
dpol=$(ls -d "$VDIR/dreamer_baseline"/PyHJ/gp/epoch_id_* 2>/dev/null | sort -t_ -k3 -n | tail -1)/policy.pth
if [ -f "$dpol" ]; then
  echo "[$(date +%H:%M:%S)] eval critic dreamer ($dpol)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/critic_ood_eval_dreamer.py --tag dreamer \
    --rssm_ckpt /data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt --policy "$dpol" \
    > "$LOG/critic_eval_dreamer.log" 2>&1
  grep "KEY critic" "$LOG/critic_eval_dreamer.log" || tail -3 "$LOG/critic_eval_dreamer.log"
fi
echo CRITIC_EVAL_DONE
