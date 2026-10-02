#!/usr/bin/env bash
# Wait for both TwoRoom checkpoints (base + jacobian, epoch 20), then run the
# requested analyses: Jacobian diagnostics (spectrum/SNR) + wall-clearance.
set -uo pipefail
cd /home/seongbin/latent/le-wm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=.venv/bin/python
LOG=/home/seongbin/latent/run_logs
CK=/data/seongbin/lewm/checkpoints
RUN=/tmp/claude-1005/-home-seongbin-latent/8f96e65f-dfab-45ae-bff3-eefdcdbbfc93/scratchpad/run_tworoom_traj.py

echo "[$(date +%H:%M:%S)] waiting for tworoom checkpoints (epoch 20)"
while [ ! -f "$CK/lewm_tworoom_base/weights_epoch_20.pt" ] || [ ! -f "$CK/lewm_tworoom_jac/weights_epoch_20.pt" ]; do
  sleep 300
done
sleep 60
echo "[$(date +%H:%M:%S)] both checkpoints present"

echo "[$(date +%H:%M:%S)] === Jacobian diagnostics ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/jac_diagnostics.py tworoom > "$LOG/tworoom_jac_diagnostics.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $PY scripts/jac_singular.py tworoom >> "$LOG/tworoom_jac_diagnostics.log" 2>&1
grep -E "===|model|baseline|jacobian|sigreg|saved" "$LOG/tworoom_jac_diagnostics.log" | grep -viE "atomic|INFO|WARN"

echo "[$(date +%H:%M:%S)] === wall-clearance trajectories (100 eps each) ==="
OBSDIST_OUT=$PWD/outputs/wallobsdist_base.npz CUDA_VISIBLE_DEVICES=0 $PY $RUN lewm_tworoom_base/weights_epoch_20.pt 100 > "$LOG/tworoom_wall_base.log" 2>&1
OBSDIST_OUT=$PWD/outputs/wallobsdist_jac.npz  CUDA_VISIBLE_DEVICES=0 $PY $RUN lewm_tworoom_jac/weights_epoch_20.pt  100 > "$LOG/tworoom_wall_jac.log" 2>&1
$PY scripts/tworoom_wall_dist.py | tee "$LOG/tworoom_wall_dist.log"
echo "TWOROOM_ANALYSIS_COMPLETE"
