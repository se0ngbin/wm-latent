#!/usr/bin/env bash
# Run the Reacher eval suite for finished WMs on another machine, at the SAME absolute paths as here.
# Per WM, on its own GPU: margin functions (reacher_margin.py, seed 0) -> zero-shot appearance OOD (reacher_ood.py)
# -> CEM raw/mlp/linear x MODE data/lewm (reacher_cem.py). The same suite was run on lewm_reacher.
# Usage: HOST=user@machine bash reacher_remote_eval.sh push|launch|status|pull|all
#   push   rsync venv + le-wm code + eval scripts + final WM weights + datasets/reacher.h5 (283 GB, resumable, skipped if already there)
#   launch start the evals on HOST (nohup + setsid, survives ssh logout); GPUS="0 1" sets one GPU per WM, in WMS order
#   status last line of every remote log
#   pull   rsync heads, CEM npz files and logs back to this machine
# Target needs: same user, /usr/bin/python3.10 (the venv's interpreter), an NVIDIA driver with EGL (MuJoCo renders), write access to
# /data/seongbin and /home/seongbin/latent, and ~300 GB free on /data.
set -euo pipefail
: "${HOST:?set HOST=user@machine}"
WMS=${WMS:-"rc_baseline rc_jacobian"}; EP=${EP:-10}; GPUS=${GPUS:-"0 1"}
LEWM=/home/seongbin/latent/le-wm; DATA=/data/seongbin/lewm
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)       # this branch's scripts (reacher_*.py are not on main)
RS=$LEWM/remote_scripts                                  # where they go on HOST (sg_point_eval adds $LEWM to sys.path itself)
LOGS=$DATA/reacher_eval_logs
R="rsync -a --partial --info=progress2"

push() {
  ssh "$HOST" "mkdir -p $LEWM $RS $DATA/datasets $DATA/checkpoints && test -x /usr/bin/python3.10 && nvidia-smi -L"
  $R --exclude wandb --exclude outputs --exclude logs --exclude __pycache__ --exclude remote_scripts "$LEWM/" "$HOST:$LEWM/"
  $R "$SRC"/reacher_margin.py "$SRC"/reacher_ood.py "$SRC"/reacher_cem.py "$SRC"/sg_point_eval.py "$HOST:$RS/"
  for wm in $WMS; do
    ssh "$HOST" "mkdir -p $DATA/checkpoints/$wm"
    $R "$DATA/checkpoints/$wm"/config.* "$DATA/checkpoints/$wm/weights_epoch_$EP.pt" "$HOST:$DATA/checkpoints/$wm/"
  done
  $R "$DATA/datasets/reacher.h5" "$HOST:$DATA/datasets/"
}

launch() {
  read -ra G <<< "$GPUS"; i=0
  for wm in $WMS; do
    gpu=${G[$((i % ${#G[@]}))]}; i=$((i + 1))
    # ponytail: one sequential chain per WM; a failed step skips the rest of that WM's chain (see its log)
    ssh "$HOST" "mkdir -p $LOGS/$wm && cd $RS && setsid nohup bash -c '
      export CUDA_VISIBLE_DEVICES=$gpu STABLEWM_HOME=$DATA LOCAL_DATASET_DIR=$DATA MUJOCO_GL=egl WM=$wm WMS=$wm EP=$EP SEED=0
      PY=$LEWM/.venv/bin/python
      \$PY reacher_margin.py > $LOGS/$wm/margin.log 2>&1 &&
      \$PY reacher_ood.py > $LOGS/$wm/ood.log 2>&1 &&
      for MODE in data lewm; do for HEAD in raw mlp linear; do
        MODE=\$MODE HEAD=\$HEAD \$PY reacher_cem.py > $LOGS/$wm/cem_\${HEAD}_\${MODE}.log 2>&1 || exit 1
      done; done && echo ALL_DONE > $LOGS/$wm/done' > $LOGS/$wm/chain.log 2>&1 < /dev/null &"
    echo "launched $wm on $HOST gpu $gpu (logs $LOGS/$wm)"
  done
}

status() { ssh "$HOST" "for f in $LOGS/*/*.log $LOGS/*/done; do [ -f \"\$f\" ] && echo \"== \$f\" && tail -c 400 \"\$f\" | tr '\r' '\n' | grep -v '^\s*\$' | tail -1; done; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"; }

pull() {
  for wm in $WMS; do
    mkdir -p "$DATA/reacher_heads/$wm" "$LOGS/$wm" "$DATA/reacher_cem"
    $R "$HOST:$DATA/reacher_heads/$wm/" "$DATA/reacher_heads/$wm/"
    $R "$HOST:$LOGS/$wm/" "$LOGS/$wm/"
    $R --include "${wm}_*.npz" --exclude '*' "$HOST:$DATA/reacher_cem/" "$DATA/reacher_cem/"
  done
  grep -h -E "REACHER_MARGIN|REACHER_OOD|FINAL" "$LOGS"/*/margin.log "$LOGS"/*/ood.log "$LOGS"/*/cem_*.log 2>/dev/null || true
}

case "${1:-}" in
  push) push ;; launch) launch ;; status) status ;; pull) pull ;; all) push && launch ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
