#!/usr/bin/env bash
# TwoRoom LEARNED wall-clearance CBF: train a margin head on the clearance field
# (per model), then eval the learned filter. Compares to gt (6.5%) and no-filter
# (base 44 / jac 52). Usage: tworoom_learned.sh [NUM_EVAL]  (default 200)
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs
MDIR=/data/seongbin/dreamer/lewm
mkdir -p "$LOG"
N=${1:-200}

train () {  # name model_dir
  local name=$1 mdl=$2
  echo "[$(date +%H:%M:%S)] train margin $name"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u train_margin_clearance_tworoom.py \
    --lewm_ckpt "$mdl/weights_epoch_20.pt" \
    --out_path "$MDIR/$mdl/margin_clearance.pt" \
    --n_states 16000 --rollout_k 5 --steps 5000 --encode_bs 64 \
    > "$LOG/tworoom_margin_train_$name.log" 2>&1
  grep -E "val_sign_acc|saved|labels:|dataset:" "$LOG/tworoom_margin_train_$name.log" | tail -3
}

eval_learned () {  # name model_dir
  local name=$1 mdl=$2
  echo "[$(date +%H:%M:%S)] eval learned $name (N=$N)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u eval.py --config-name=tworoom solver=safe_tworoom \
    solver.safe_mode=learned solver.margin_ckpt="$MDIR/$mdl/margin_clearance.pt" \
    policy="$mdl/weights_epoch_20.pt" world.max_episode_steps=60 \
    eval.num_eval=$N eval.goal_offset_steps=25 eval.eval_budget=50 \
    plan_config.horizon=5 plan_config.action_block=5 plan_config.receding_horizon=5 \
    hydra.run.dir="outputs/tworoom_learned_$name" \
    > "$LOG/tworoom_learned_$name.log" 2>&1
  echo -n "  $name: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/tworoom_learned_$name.log" | tr '\n' ' '
  echo -n " | "; grep -oE "fallback_rate=[0-9.]+" "$LOG/tworoom_learned_$name.log" | tail -1
}

train base lewm_tworoom_base
train jac  lewm_tworoom_jac
eval_learned base lewm_tworoom_base
eval_learned jac  lewm_tworoom_jac
echo TWOROOM_LEARNED_DONE
