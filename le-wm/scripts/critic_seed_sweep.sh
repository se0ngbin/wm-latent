#!/usr/bin/env bash
# Multi-seed sweep for the value-function OOD table (in-dist + retrained), uniform
# naming under _value_ood/_sweep so aggregation is trivial. 4 models x
# {in-dist(red), purple, diamond, rot90} x seeds -> 16 critics/seed. Each ~4.5 min,
# 2 at a time on 2 GPUs. Reuses the seed-0 per-appearance margins as the reward for
# all seeds (we report the CRITIC's seed variance; margin variance is in its own table).
# Usage: critic_seed_sweep.sh "0 1 2"
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
VDIR=/data/seongbin/dreamer/lewm/_value_ood; MDIR=$VDIR/margins; SW=$VDIR/_sweep; LOG=/home/seongbin/latent/run_logs
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
mkdir -p "$SW" "$LOG"
SEEDS=${1:-"0 1 2"}
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
declare -A ENVV=( [purple]="CRITIC_OBST_COLOR=purple" [diamond]="CRITIC_OBST_SHAPE=diamond" [rot90]="CRITIC_ROT_DEG=90" )

run_job() {  # kind tag cond seed gpu     (kind: indist|retr ; cond ignored for indist)
  local kind=$1 tag=$2 cond=$3 seed=$4 gpu=$5 w=${WM[$2]:-}
  local name lg envv=""
  if [ "$kind" = indist ]; then name="${tag}_s${seed}"; else name="${tag}_${cond}_s${seed}"; envv="${ENVV[$cond]}"; fi
  local logdir="$SW/$name"; lg="$LOG/sweep_${name}.log"
  local pol; pol=$(ls -d "$logdir"/PyHJ/gp/epoch_id_* 2>/dev/null | head -1)
  [ -n "$pol" ] && [ -f "$pol/policy.pth" ] && { echo "  cached $name"; return; }
  if [ "$tag" = dreamer ]; then
    if [ "$kind" = indist ]; then
      env $envv CUDA_VISIBLE_DEVICES=$gpu $PY scripts/wm_ddpg.py --wm_backend dreamer \
        --rssm-ckpt "$RSSM" --logdir "$logdir" --seed "$seed" > "$lg" 2>&1
    else
      env $envv CUDA_VISIBLE_DEVICES=$gpu $PY scripts/wm_ddpg.py --wm_backend dreamer \
        --rssm-ckpt "$RSSM" --dreamer_margin_ckpt "$MDIR/dreamer_${cond}.pt" \
        --logdir "$logdir" --seed "$seed" > "$lg" 2>&1
    fi
  else
    local margin
    [ "$kind" = indist ] && margin="/data/seongbin/dreamer/lewm/$w/margin_heads.pt" || margin="$MDIR/${tag}_${cond}.pt"
    env $envv CUDA_VISIBLE_DEVICES=$gpu $PY scripts/wm_ddpg.py --wm_backend lewm \
      --lewm_run_name "$w" --lewm_ckpt_path /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
      --lewm_margin_ckpt "$margin" --logdir "$logdir" --seed "$seed" > "$lg" 2>&1
  fi
  [ $? -eq 0 ] && echo "  done $name" || echo "  FAILED $name"
}

# Build job list
JOBS=()
for s in $SEEDS; do
  for tag in baseline jacobian jacpull dreamer; do
    JOBS+=("indist:$tag:red:$s")
    for cond in purple diamond rot90; do JOBS+=("retr:$tag:$cond:$s"); done
  done
done
echo "[$(date +%H:%M:%S)] sweep: ${#JOBS[@]} critics on seeds [$SEEDS]"
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  a=${JOBS[$i]}; b=${JOBS[$((i+1))]:-}
  IFS=: read ka ta ca sa <<< "$a"; run_job "$ka" "$ta" "$ca" "$sa" 0 &
  if [ -n "$b" ]; then IFS=: read kb tb cb sb <<< "$b"; run_job "$kb" "$tb" "$cb" "$sb" 1 & fi
  wait
  i=$((i+2))
done
echo CRITIC_SWEEP_TRAIN_DONE
