#!/usr/bin/env bash
# C3: closed-loop no-filter collision under obstacle-appearance shift (Dubins).
# WM frozen; only the RENDERED obstacle changes (color/shape) — dynamics, geometry,
# _collides, and goal are untouched, so the task is identical and only the planner's
# OBSERVATION goes OOD. Tests whether EMERGENT obstacle avoidance (no safety filter)
# survives appearance shift, per encoder. Rotation omitted (rot90'ing the view breaks
# the obs<->action correspondence in closed loop). Usage: ood_c3_dubins.sh [N]
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs; mkdir -p "$LOG"
N=${1:-100}

declare -A ENC=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )

run () {  # enc_name model  cond  env-assignments...
  local name=$1 mdl=$2 cond=$3; shift 3
  echo "[$(date +%H:%M:%S)] $name / $cond (N=$N)"
  env "$@" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=dubins solver=cem \
    policy="$mdl/weights_epoch_50.pt" world.max_episode_steps=40 \
    eval.num_eval=$N eval.eval_budget=30 \
    hydra.run.dir="outputs/c3_${name}_${cond}" \
    > "$LOG/ood_c3_${name}_${cond}.log" 2>&1
  echo -n "  $name/$cond: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/ood_c3_${name}_${cond}.log" | tr '\n' ' '; echo
}

for name in baseline jacobian jacpull; do
  mdl=${ENC[$name]}
  run "$name" "$mdl" red
  run "$name" "$mdl" purple  DUBINS_OBST_COLOR=purple
  run "$name" "$mdl" diamond DUBINS_OBST_SHAPE=square
done
echo "OOD_C3_DONE"
