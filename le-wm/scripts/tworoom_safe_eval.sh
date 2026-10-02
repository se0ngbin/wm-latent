#!/usr/bin/env bash
# TwoRoom gt safe-filter eval ('contact with wall = unsafe'). Compares, per model,
# no-filter CEM vs the gt wall-clearance CBF (SafeTwoRoomCEMSolver). Reports
# success_rate + collision_rate (env _collided wall-contact flag) + fallback.
# Usage: tworoom_safe_eval.sh [NUM_EVAL]   (default 200; use small for smoke)
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"
N=${1:-200}

run () {  # name  model_dir  solver
  local name=$1 mdl=$2 solver=$3
  echo "[$(date +%H:%M:%S)] $name (solver=$solver, N=$N)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=tworoom solver=$solver \
    policy="$mdl/weights_epoch_20.pt" \
    world.max_episode_steps=60 \
    eval.num_eval=$N eval.goal_offset_steps=25 eval.eval_budget=50 \
    plan_config.horizon=5 plan_config.action_block=5 plan_config.receding_horizon=5 \
    hydra.run.dir="outputs/tworoom_safe_${name}" \
    > "$LOG/tworoom_safe_${name}.log" 2>&1
  echo -n "  $name: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/tworoom_safe_${name}.log" | tr '\n' ' '
  echo -n " | "; grep -oE "fallback_rate=[0-9.]+" "$LOG/tworoom_safe_${name}.log" | tail -1
}

run base_nofilter  lewm_tworoom_base cem
run base_gt        lewm_tworoom_base safe_tworoom
run jac_nofilter   lewm_tworoom_jac  cem
run jac_gt         lewm_tworoom_jac  safe_tworoom
echo TWOROOM_SAFE_DONE
