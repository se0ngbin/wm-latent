#!/usr/bin/env bash
# Wait for the two schedule-fixed training runs to finish, then eval each final
# (fully-annealed) checkpoint with the LE-WM planner. Launched in background.
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm
PY=.venv/bin/python
CKPT="$STABLEWM_HOME/checkpoints"
LOG=/home/seongbin/latent/run_logs

PUSHT_PID="${1:-423224}"
DUBINS_PID="${2:-423225}"

echo "[$(date +%H:%M:%S)] waiting for training PIDs $PUSHT_PID (pushT), $DUBINS_PID (dubins)"
while kill -0 "$PUSHT_PID" 2>/dev/null || kill -0 "$DUBINS_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] both training runs exited"

pick() { ls "$CKPT/$1"/weights_epoch_*.pt 2>/dev/null | sort -V | tail -1; }

PUSHT_CK="$(pick sigreg_pusht_sched10)"
DUBINS_CK="$(pick sigreg_dubins_sched50)"
echo "pushT ckpt:  ${PUSHT_CK:-MISSING}"
echo "dubins ckpt: ${DUBINS_CK:-MISSING}"

if [ -n "$PUSHT_CK" ]; then
  echo "[$(date +%H:%M:%S)] eval pushT $(basename "$PUSHT_CK")"
  CUDA_VISIBLE_DEVICES=0 $PY eval.py --config-name=pusht \
    policy="sigreg_pusht_sched10/$(basename "$PUSHT_CK")" \
    hydra.run.dir=outputs/eval_sigreg_pusht_sched10 \
    > "$LOG/autoeval_pusht_sched10.log" 2>&1
  echo "[$(date +%H:%M:%S)] pushT eval done (exit $?)"
fi

if [ -n "$DUBINS_CK" ]; then
  echo "[$(date +%H:%M:%S)] eval dubins $(basename "$DUBINS_CK") (200 eps)"
  CUDA_VISIBLE_DEVICES=0 $PY eval.py --config-name=dubins \
    policy="sigreg_dubins_sched50/$(basename "$DUBINS_CK")" eval.num_eval=200 \
    hydra.run.dir=outputs/eval_sigreg_dubins_sched50 \
    > "$LOG/autoeval_dubins_sched50.log" 2>&1
  echo "[$(date +%H:%M:%S)] dubins eval done (exit $?)"
fi

echo "AUTO_EVAL_COMPLETE"
