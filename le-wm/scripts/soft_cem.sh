#!/usr/bin/env bash
# Soft-cost CEM: goal + w * sum_t relu(-h(z_t)) on imagined latents (HJ margin head), sg25clean protocol
# (same eval.py overrides as raw_cem_ood.sh). Usage: soft_cem.sh <gpu> <wm_dir> <head.pt> <tag> <w> [scen ...]
# Optional env CONST=<c> adds c * 1[any violation]; BS = envs per CEM batch (16; 1 is launch-bound when CPU is shared). scen in red purple diamond fullrot90 (default red).
# Retries on CUDA OOM (the GPUs are shared with training jobs): 5 min backoff, up to 6 tries.
# Edit by writing a new file and mv-ing it over this one: running copies keep reading the old inode.
set -uo pipefail
cd "$(dirname "$0")/.."
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
# Cap CPU threads: with default (=all cores) threads, 6 concurrent evals on a loaded box spin in futex waits and
# never get through setup (observed: 0 progress in 60 min at load ~80 on 24 cores).
export PYTHONUNBUFFERED=1  # print() progress (goals, CEM solve time) reaches the log live
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4}
LOG=/home/seongbin/latent/run_logs; PY=/home/seongbin/latent/le-wm/.venv/bin/python
GPU=$1 WM=$2 HEAD=$3 TAG=$4 W=$5; shift 5
C=${CONST:-0}; SUF=w${W}; [ "$C" != 0 ] && SUF=${SUF}_c${C}   # CONST env: lexicographic flat cost
[ "${REC:-25}" != 25 ] && SUF=${SUF}_rec${REC}             # REC env: replan every REC steps (default 25 = open loop)
declare -A ENVS=([red]= [purple]=DUBINS_OBST_COLOR=purple [diamond]=DUBINS_OBST_SHAPE=square [fullrot90]=DUBINS_ROT_DEG=90)
for scen in "${@:-red}"; do
  lg="$LOG/softcem_${TAG}_${SUF}_${scen}.log"
  for try in 1 2 3 4 5 6; do
    grep -q "'collision_rate'" "$lg" 2>/dev/null && break
    env ${ENVS[$scen]} CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --config-name=dubins solver=soft_cem \
      solver.margin_ckpt="$HEAD" solver.unsafe_weight=$W solver.unsafe_const=$C solver.batch_size=${BS:-16} policy="$WM/weights_epoch_50.pt" \
      eval.num_eval=200 eval.goal_offset_steps=25 eval.goal_at_episode_end=false eval.eval_budget=50 \
      plan_config.horizon=25 plan_config.action_block=1 plan_config.receding_horizon=${REC:-25} \
      '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' +eval.goal_avoid_margin=0.05 \
      +eval.goal_near_margin=0.5 +eval.goal_clean_source=true hydra.run.dir="outputs/softcem_${TAG}_${SUF}_${scen}" > "$lg" 2>&1
    grep -q OutOfMemoryError "$lg" || break
    echo "$(date +%H:%M) $TAG $SUF $scen OOM (try $try), retrying in 5 min"; sleep 300
  done
  echo "$(date +%H:%M) $TAG $SUF $scen $(grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$lg" | tr '\n' ' ')"
done
