#!/usr/bin/env bash
# TwoRoom learned_qp: min-intervention CBF filter (HardNet++ damped projection) on
# the EXECUTED block only, 1-step, re-grounded on the real obs each step (receding
# horizon = 1). Reuses the margin heads trained by tworoom_learned.sh. Compares to
# no-filter (base 22.5% coll target), gt (6.5%), and learned-grid (neutral).
# Usage: tworoom_qp.sh [NUM_EVAL]  (default 200; small N for smoke)
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
MDIR=/data/seongbin/dreamer/lewm
mkdir -p "$LOG"
N=${1:-200}

eval_qp () {  # name model_dir
  local name=$1 mdl=$2
  echo "[$(date +%H:%M:%S)] eval learned_qp $name (N=$N)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=tworoom solver=safe_tworoom \
    solver.safe_mode=learned_qp solver.margin_ckpt="$MDIR/$mdl/margin_clearance.pt" \
    solver.receding_horizon=1 solver.qp_iters=8 \
    policy="$mdl/weights_epoch_20.pt" world.max_episode_steps=60 \
    eval.num_eval=$N eval.goal_offset_steps=25 eval.eval_budget=50 \
    plan_config.horizon=5 plan_config.action_block=5 plan_config.receding_horizon=1 \
    hydra.run.dir="outputs/tworoom_qp_$name" \
    > "$LOG/tworoom_qp_$name.log" 2>&1
  echo -n "  $name: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/tworoom_qp_$name.log" | tr '\n' ' '
  echo -n " | "; grep -oE "fallback_rate=[0-9.]+" "$LOG/tworoom_qp_$name.log" | tail -1
}

eval_qp base lewm_tworoom_base
eval_qp jac  lewm_tworoom_jac
echo TWOROOM_QP_DONE
