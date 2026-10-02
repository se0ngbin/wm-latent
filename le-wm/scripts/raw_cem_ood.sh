#!/usr/bin/env bash
# Raw CEM, NO filter, closed-loop under OOD appearance shift, for the 3 LEWM models.
# WM frozen; only the deployment world's rendering shifts (color/shape/full-world rot).
#
# PROTOCOL: the model-DIFFERENTIATING sg25clean protocol (matches
# scripts/eval_twoway50_sg25clean.sh) — goal_offset=25 mid-trajectory sub-goals,
# annulus goals around the obstacles, clean-success source, horizon=25,
# receding_horizon=25, budget=50. On this protocol the models actually differ on
# safety (sigreg ~50% collision vs jacobian ~22.5%). The old easy dataset-replay
# protocol (goal_at_episode_end, 12-step hops) gave ~8% for everyone and did NOT
# discriminate.  Reports collision% / success%. Usage: raw_cem_ood.sh [N]
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
LOG=/home/seongbin/latent/run_logs; mkdir -p "$LOG"
N=${1:-200}
GPU=${2:-0}   # run serially on ONE GPU so we don't contend with the critic batch
PY=.venv/bin/python
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )

run () {  # tag scenario env-assignments...
  local tag=$1 scen=$2; shift 2; local w=${WM[$tag]}
  local lg="$LOG/rawcem_sg25_${tag}_${scen}.log"
  if grep -q "2x2 \[" "$lg" 2>/dev/null; then
    echo -n "  $tag/$scen (cached): "; grep -oE "safe_success=[0-9.]+ unsafe_success=[0-9.]+ safe_fail=[0-9.]+ unsafe_fail=[0-9.]+" "$lg" | tail -1; return
  fi
  env "$@" CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --config-name=dubins \
    policy="$w/weights_epoch_50.pt" \
    eval.num_eval=$N \
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
    hydra.run.dir="outputs/rawcem_sg25_${tag}_${scen}" \
    > "$LOG/rawcem_sg25_${tag}_${scen}.log" 2>&1
  echo -n "  $tag/$scen: "
  grep -oE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$lg" | tr '\n' ' '
  echo -n " | "; grep -oE "safe_success=[0-9.]+ unsafe_success=[0-9.]+ safe_fail=[0-9.]+ unsafe_fail=[0-9.]+" "$lg" | tail -1; echo
}

# Serial: 4 conditions x 3 models, one at a time on GPU $GPU.
for tag in baseline jacobian jacpull; do
  echo "[$(date +%H:%M:%S)] === $tag ==="
  run "$tag" red
  run "$tag" purple    DUBINS_OBST_COLOR=purple
  run "$tag" diamond   DUBINS_OBST_SHAPE=square
  run "$tag" fullrot90 DUBINS_ROT_DEG=90
done
echo RAW_CEM_OOD_DONE
