#!/usr/bin/env bash
# On-distribution margin retrain + re-eval. Retrains the HJ margin head with
# full-horizon imagined rollouts (rollout_k=25) and near-obstacle start states
# (the region where the CEM planner dwells and the imagined-margin fallback
# spikes at 12-15%). Then re-evals through the safe planner (hard-map) to check
# whether fallback drops and the learned CBF stops being net-harmful.
set -uo pipefail
LOG=/home/seongbin/latent/run_logs
mkdir -p "$LOG"
CBF=/home/seongbin/latent/latent_cbf
LEWM=/home/seongbin/latent/le-wm

retrain () {  # name  lewm_ckpt_dir
  local name=$1 mdl=$2
  echo "[$(date +%H:%M:%S)] retrain margin (on-dist) $name"
  cd "$CBF"
  STABLEWM_HOME=/data/seongbin/lewm CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
    src/latent_cbf/scripts/train_margin_hj_lewm.py \
    --lewm_ckpt "$mdl/weights_epoch_50.pt" \
    --out_path "/data/seongbin/dreamer/lewm/$mdl/margin_hj_ondist.pt" \
    --n_states 20000 --rollout_k 25 --near_obstacle_frac 0.5 --steps 5000 \
    > "$LOG/margin_ondist_train_$name.log" 2>&1
  grep -E "val_sign_acc|saved|near-obstacle|dataset" "$LOG/margin_ondist_train_$name.log" | tail -4
}

reval () {  # name  lewm_ckpt_dir  map_candidates
  local name=$1 mdl=$2 mapc=$3
  echo "[$(date +%H:%M:%S)] eval $name (map_candidates=$mapc)"
  cd "$LEWM"
  export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=dubins solver=safe_cem \
    solver.safe_mode=learned \
    solver.margin_ckpt="/data/seongbin/dreamer/lewm/$mdl/margin_hj_ondist.pt" \
    solver.map_candidates=$mapc solver.batch_size=16 \
    policy="$mdl/weights_epoch_50.pt" \
    eval.num_eval=200 eval.goal_offset_steps=25 eval.goal_at_episode_end=false eval.eval_budget=50 \
    plan_config.horizon=25 plan_config.action_block=1 plan_config.receding_horizon=25 \
    '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' \
    +eval.goal_avoid_margin=0.05 +eval.goal_near_margin=0.5 +eval.goal_clean_source=true \
    hydra.run.dir="outputs/margin_ondist_${name}_map${mapc}" \
    > "$LOG/margin_ondist_eval_${name}_map${mapc}.log" 2>&1
  echo -n "  $name map=$mapc: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/margin_ondist_eval_${name}_map${mapc}.log" | tr '\n' ' '
  echo -n " | "; grep -oE "fallback_rate=[0-9.]+" "$LOG/margin_ondist_eval_${name}_map${mapc}.log" | tail -1
}

retrain jacobian jacobian_w1_dubins
retrain jacpull  lewm_dubins_jacpull50
reval jacobian jacobian_w1_dubins true
reval jacpull  lewm_dubins_jacpull50 true
echo MARGIN_ONDIST_DONE
