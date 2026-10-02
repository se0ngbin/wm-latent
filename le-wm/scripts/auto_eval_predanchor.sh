#!/usr/bin/env bash
# Wait for the pred-anchor checkpoint, then: (1) correct-protocol planning eval,
# (2) safe/unsafe linear probe (baseline vs jacobian vs pred_anchor),
# (3) Jacobian diagnostics on the pred-anchor encoder.
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=.venv/bin/python; LOG=/home/seongbin/latent/run_logs
CK=/data/seongbin/lewm/checkpoints/lewm_dubins_predanchor/weights_epoch_50.pt

echo "[$(date +%H:%M:%S)] waiting for pred-anchor checkpoint"
while [ ! -f "$CK" ]; do sleep 300; done
sleep 60
echo "[$(date +%H:%M:%S)] checkpoint present"

echo "[$(date +%H:%M:%S)] === planning eval (sg25clean, 200 eps) ==="
CUDA_VISIBLE_DEVICES=0 $PY eval.py --config-name=dubins \
  policy=lewm_dubins_predanchor/weights_epoch_50.pt \
  eval.num_eval=200 eval.goal_offset_steps=25 eval.goal_at_episode_end=false eval.eval_budget=50 \
  plan_config.horizon=25 plan_config.action_block=1 plan_config.receding_horizon=25 \
  '+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]' \
  +eval.goal_avoid_margin=0.05 +eval.goal_near_margin=0.5 +eval.goal_clean_source=true \
  hydra.run.dir=outputs/dubins_sg25clean200_predanchor > "$LOG/eval_dubins_sg25clean200_predanchor.log" 2>&1
echo "  $(grep -aoE "'success_rate': [0-9.]+|'collision_rate': [0-9.]+" "$LOG/eval_dubins_sg25clean200_predanchor.log" | head -2 | tr '\n' ' ')"

echo "[$(date +%H:%M:%S)] === safe/unsafe probe (baseline vs jacobian vs pred_anchor) ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/safety_probe.py > "$LOG/safety_probe_predanchor.log" 2>&1
grep -aE "AUC|R\^2|balanced set" "$LOG/safety_probe_predanchor.log" | grep -aviE "atomic|INFO|WARN"

echo "[$(date +%H:%M:%S)] === Jacobian diagnostics on pred_anchor encoder ==="
CUDA_VISIBLE_DEVICES=0 $PY - <<'PYEOF' 2>&1 | grep -aE "pred_anchor|baseline|jacobian|model"
import os,sys; sys.path.insert(0,"scripts"); sys.argv=["x","dubins"]
import jac_diagnostics as J
# add pred_anchor to the model set and rerun the norm/cond summary
J.CFG["models"]["pred_anchor"]="lewm_dubins_predanchor"
J.main()
PYEOF
echo "PREDANCHOR_EVAL_COMPLETE"
