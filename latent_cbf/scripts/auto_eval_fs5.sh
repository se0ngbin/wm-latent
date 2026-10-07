#!/usr/bin/env bash
# Wait for the fs=5 dubins retrain to finish, then run the ONLINE LE-WM planner
# eval (8x5: horizon 8, action_block 5) and report success + collision.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf
export STABLEWM_HOME=/data/seongbin/lewm MUJOCO_GL=egl
PY=.venv/bin/python
CKDIR="$STABLEWM_HOME/checkpoints/sigreg_dubins_fs5"
LOG=/home/seongbin/latent/run_logs

FS5_PID="${1:-557772}"
echo "[$(date +%H:%M:%S)] waiting for fs5 training PID $FS5_PID"
while kill -0 "$FS5_PID" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] fs5 training exited"

CK=$(ls "$CKDIR"/weights_epoch_*.pt 2>/dev/null | sort -V | tail -1)
echo "fs5 ckpt: ${CK:-MISSING}"
[ -z "$CK" ] && { echo "no fs5 checkpoint; aborting"; exit 1; }

echo "[$(date +%H:%M:%S)] online eval (8x5) on $(basename "$CK")"
CUDA_VISIBLE_DEVICES=0 $PY src/latent_cbf/scripts/eval_lewm_planner.py \
  --episodes 50 --horizon 8 --num_samples 300 --n_iters 5 --topk 30 \
  --ckpt "sigreg_dubins_fs5/$(basename "$CK")" \
  > "$LOG/autoeval_fs5_online.log" 2>&1
echo "[$(date +%H:%M:%S)] fs5 online eval done (exit $?)"
echo "AUTO_EVAL_FS5_COMPLETE"
