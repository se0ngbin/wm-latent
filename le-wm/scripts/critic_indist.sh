#!/usr/bin/env bash
# Value-function OOD, Phase 1: train the in-dist (red) reachability critics (gp)
# from the instantaneous margin, per model. ~33 min/critic. Gives in-dist + (via
# separate eval on shifted feats) zero-shot columns. JEPA {baseline,jacobian,
# jac+pull} + Dreamer baseline.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs; VDIR=/data/seongbin/dreamer/lewm/_value_ood
mkdir -p "$LOG" "$VDIR"
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
for tag in baseline jacobian jacpull; do
  w=${WM[$tag]}
  echo "[$(date +%H:%M:%S)] critic in-dist jepa/$tag ($w)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/wm_ddpg.py --wm_backend lewm --lewm_run_name "$w" \
    --lewm_ckpt_path /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
    --lewm_margin_ckpt /data/seongbin/dreamer/lewm/$w/margin_heads.pt \
    --logdir "$VDIR/jepa_$tag" --seed 0 > "$LOG/critic_indist_jepa_$tag.log" 2>&1 \
    && echo "  done jepa/$tag" || echo "  FAILED jepa/$tag"
done
echo "[$(date +%H:%M:%S)] critic in-dist dreamer/baseline"
CUDA_VISIBLE_DEVICES=0 $PY scripts/wm_ddpg.py --wm_backend dreamer \
  --rssm-ckpt /data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt \
  --logdir "$VDIR/dreamer_baseline" --seed 0 > "$LOG/critic_indist_dreamer.log" 2>&1 \
  && echo "  done dreamer" || echo "  FAILED dreamer"
echo CRITIC_INDIST_DONE
