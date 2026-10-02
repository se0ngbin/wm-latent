#!/usr/bin/env bash
# Execution-only safe-mapping test: CEM optimizes the goal cost over RAW candidates
# (map_candidates=false), safety is applied only to the executed prefix. Tests the
# hypothesis that safe-by-construction CEM is net-harmful because the optimizer
# exploits the learned margin's false-negatives. Reactive-style filter should instead
# REDUCE collision vs the no-filter CEM (jac 22.5%, combo 22.x%), toward the
# DDPG-actor's 7.5% / 1.6%.
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"

run () {  # name  model_dir  margin_ckpt
  local name=$1 mdl=$2 mck=$3
  echo "[$(date +%H:%M:%S)] $name"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=dubins solver=safe_cem \
    solver.safe_mode=learned solver.margin_ckpt="$mck" \
    solver.map_candidates=false solver.batch_size=16 \
    policy="$mdl/weights_epoch_50.pt" \
    eval.num_eval=200 eval.goal_offset_steps=25 eval.goal_at_episode_end=false eval.eval_budget=50 \
    plan_config.horizon=25 plan_config.action_block=1 plan_config.receding_horizon=25 \
    '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' \
    +eval.goal_avoid_margin=0.05 +eval.goal_near_margin=0.5 +eval.goal_clean_source=true \
    hydra.run.dir="outputs/cem_execfilter_$name" \
    > "$LOG/cem_execfilter_$name.log" 2>&1
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/cem_execfilter_$name.log" | tr '\n' ' ' | sed "s/^/  $name: /"; echo
}

run jacobian jacobian_w1_dubins /data/seongbin/dreamer/lewm/jacobian_w1_dubins/margin_hj_imag.pt
run jacpull  lewm_dubins_jacpull50 /data/seongbin/dreamer/lewm/lewm_dubins_jacpull50/margin_hj_imag.pt
echo CEM_EXECFILTER_DONE
