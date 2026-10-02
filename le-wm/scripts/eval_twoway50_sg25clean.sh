#!/usr/bin/env bash
# Correct-protocol eval of the dubins two-way ablation (50-epoch runs):
# sg25 clean-source annulus goals, horizon=25, receding_horizon=25, budget=50.
# Matches outputs/dubins_sg25clean200_fs1h25rec25 (sigreg baseline: 62.5/50.0).
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm
export LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=.venv/bin/python
LOG=/home/seongbin/latent/run_logs

run_eval() {  # $1=name $2=gpu
  echo "[$(date +%H:%M:%S)] eval $1 (gpu $2)"
  CUDA_VISIBLE_DEVICES=$2 $PY eval.py --config-name=dubins \
    policy="lewm_dubins_$1/weights_epoch_50.pt" \
    eval.num_eval=200 \
    eval.goal_offset_steps=25 \
    eval.goal_at_episode_end=false \
    eval.eval_budget=50 \
    plan_config.horizon=25 \
    plan_config.action_block=1 \
    plan_config.receding_horizon=25 \
    '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' \
    +eval.goal_avoid_margin=0.05 \
    +eval.goal_near_margin=0.5 \
    +eval.goal_clean_source=true \
    hydra.run.dir="outputs/dubins_sg25clean200_$1" \
    > "$LOG/eval_dubins_sg25clean200_$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] $1 done (exit $?)"
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/eval_dubins_sg25clean200_$1.log" | head -2
}

run_eval encreg50 0 &
run_eval predreg50 1 &
wait
run_eval twoway50 0
echo "SG25CLEAN_EVAL_COMPLETE"
