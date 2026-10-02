#!/usr/bin/env bash
# Control for the learned_qp comparison: no-filter CEM at receding_horizon=1 (same
# re-grounding cadence as the QP filter), so any QP gain is attributable to the
# filter, not to finer replanning. Usage: tworoom_nofilter_rec1.sh [NUM_EVAL]
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"
N=${1:-200}

run () {  # name model_dir
  local name=$1 mdl=$2
  echo "[$(date +%H:%M:%S)] no-filter rec=1 $name (N=$N)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=tworoom solver=cem \
    policy="$mdl/weights_epoch_20.pt" world.max_episode_steps=60 \
    eval.num_eval=$N eval.goal_offset_steps=25 eval.eval_budget=50 \
    plan_config.horizon=5 plan_config.action_block=5 plan_config.receding_horizon=1 \
    hydra.run.dir="outputs/tworoom_nf1_$name" \
    > "$LOG/tworoom_nf1_$name.log" 2>&1
  echo -n "  $name: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/tworoom_nf1_$name.log" | tr '\n' ' '; echo
}

run base lewm_tworoom_base
run jac  lewm_tworoom_jac
echo TWOROOM_NF1_DONE
