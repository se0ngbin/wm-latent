#!/usr/bin/env bash
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
for d in 0.1 0.15; do
  echo "[$(date +%H:%M:%S)] h_shift=$d"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=dubins solver=safe_cem \
    solver.safe_mode=learned solver.margin_ckpt=/data/seongbin/dreamer/lewm/pixel_lipschitz_dubins/margin_hj_imag.pt \
    solver.batch_size=16 solver.h_shift=$d \
    policy="pixel_lipschitz_dubins/weights_epoch_50.pt" \
    eval.num_eval=50 eval.goal_offset_steps=25 eval.goal_at_episode_end=false eval.eval_budget=50 \
    plan_config.horizon=25 plan_config.action_block=1 plan_config.receding_horizon=4 \
    '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' \
    +eval.goal_avoid_margin=0.05 +eval.goal_near_margin=0.5 +eval.goal_clean_source=true \
    hydra.run.dir="outputs/dubins_sg25clean50_rec4_pixlip_shift$d" \
    > "$LOG/eval_rec4_pixlip_shift$d.log" 2>&1
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/eval_rec4_pixlip_shift$d.log" | sed "s/^/shift=$d /"
done
echo HSHIFT_SWEEP_DONE
