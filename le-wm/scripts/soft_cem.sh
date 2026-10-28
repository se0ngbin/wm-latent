#!/usr/bin/env bash
# Soft-cost CEM: goal + w * sum_t relu(-h(z_t)) on imagined latents (HJ margin head), sg25clean protocol
# (same eval.py overrides as raw_cem_ood.sh). Usage: soft_cem.sh <gpu> <wm_dir> <head.pt> <tag> <w> [scen ...]
# Optional env CONST=<c> adds c * 1[any violation]. scen in red purple diamond fullrot90 (default red).
set -uo pipefail
cd "$(dirname "$0")/.."
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs; PY=/home/seongbin/latent/le-wm/.venv/bin/python
GPU=$1 WM=$2 HEAD=$3 TAG=$4 W=$5; shift 5
C=${CONST:-0}; SUF=w${W}; [ "$C" != 0 ] && SUF=${SUF}_c${C}   # CONST env: lexicographic flat cost
declare -A ENVS=([red]= [purple]=DUBINS_OBST_COLOR=purple [diamond]=DUBINS_OBST_SHAPE=square [fullrot90]=DUBINS_ROT_DEG=90)
for scen in "${@:-red}"; do
  lg="$LOG/softcem_${TAG}_${SUF}_${scen}.log"
  grep -q "'collision_rate'" "$lg" 2>/dev/null && continue
  env ${ENVS[$scen]} CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --config-name=dubins solver=soft_cem \
    solver.margin_ckpt="$HEAD" solver.unsafe_weight=$W solver.unsafe_const=$C policy="$WM/weights_epoch_50.pt" \
    eval.num_eval=200 eval.goal_offset_steps=25 eval.goal_at_episode_end=false eval.eval_budget=50 \
    plan_config.horizon=25 plan_config.action_block=1 plan_config.receding_horizon=25 \
    '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' +eval.goal_avoid_margin=0.05 \
    +eval.goal_near_margin=0.5 +eval.goal_clean_source=true hydra.run.dir="outputs/softcem_${TAG}_${SUF}_${scen}" > "$lg" 2>&1
  echo "$(date +%H:%M) $TAG $SUF $scen $(grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$lg" | tr '\n' ' ')"
done
