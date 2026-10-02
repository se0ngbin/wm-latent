#!/usr/bin/env bash
# RETRAINED value-function column, Dreamer baseline: train a reachability critic
# per appearance with (a) a per-appearance dreamer margin (--dreamer_margin_ckpt)
# and (b) reset-seed images re-rendered under that appearance (CRITIC_OBST_*).
# Waits for the JEPA retrained batch to finish first (shared 2 GPUs).
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
VDIR=/data/seongbin/dreamer/lewm/_value_ood; MDIR=$VDIR/margins; LOG=/home/seongbin/latent/run_logs
JEPA_MARK="$LOG/critic_retrained_batch.log"
declare -A ENVV=( [purple]="CRITIC_OBST_COLOR=purple" [diamond]="CRITIC_OBST_SHAPE=diamond" [rot90]="CRITIC_ROT_DEG=90" )

# Wait until the JEPA retrained batch is done, so we don't over-subscribe GPUs.
echo "[$(date +%H:%M:%S)] waiting for JEPA retrained batch to finish..."
until grep -q "CRITIC_RETRAINED_DONE" "$JEPA_MARK" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] JEPA done; starting dreamer retrained critics."

train_one() {  # cond gpu
  local cond=$1 gpu=$2
  local margin="$MDIR/dreamer_${cond}.pt"
  [ -f "$margin" ] || { echo "  MISSING $margin"; return 1; }
  echo "[$(date +%H:%M:%S)] train dreamer critic /$cond (gpu $gpu, env ${ENVV[$cond]})"
  env ${ENVV[$cond]} CUDA_VISIBLE_DEVICES=$gpu $PY scripts/wm_ddpg.py --wm_backend dreamer \
    --rssm-ckpt "$RSSM" --dreamer_margin_ckpt "$margin" \
    --logdir "$VDIR/dreamer_${cond}" --seed 0 \
    > "$LOG/critic_retrained_dreamer_${cond}.log" 2>&1 \
    && echo "  done dreamer/$cond" || echo "  FAILED dreamer/$cond (exit $?)"
}

train_one purple  0 &
train_one diamond 1 &
wait
train_one rot90   0
echo CRITIC_RETRAINED_DREAMER_DONE
