#!/usr/bin/env bash
# LE-WM training driver. One preset per world model; each is a `train.py` run with
# a regularizer override. Replaces the old per-experiment train_*.sh + the sweep
# harness. Presets are dispatched round-robin across GPUS (serial per GPU).
#
# Usage (from le-wm/):
#   scripts/run.sh                       # every preset
#   scripts/run.sh jacobian_dubins no_reg
#   GPUS=0 scripts/run.sh sigreg_only    # pin to one GPU
#   EPOCHS=10 EXTRA_ARGS="loader.batch_size=64" scripts/run.sh jacobian_pusht
#
# Env: STABLEWM_HOME (data+ckpt root), GPUS ("0,1"), EPOCHS (override per-preset
# default), EXTRA_ARGS (appended verbatim to every run).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
: "${STABLEWM_HOME:=/data/seongbin/lewm}"; export STABLEWM_HOME
: "${GPUS:=0,1}"
: "${EXTRA_ARGS:=}"
LOG=/home/seongbin/latent/run_logs; mkdir -p "$LOG"

# preset -> "data|epochs|hydra-override"   (override "" = sigreg-only baseline)
declare -A RUNS=(
  [sigreg_only]="pusht|100|"
  [no_reg]="pusht|100|~loss.regularizers.sigreg"
  [jacobian_dubins]="dubins|50|+loss.regularizers.jacobian={weight: 0.1, kwargs: {target_L: 1.0, n_probes: 1}}"
  [jacobian_w1_dubins]="dubins|50|+loss.regularizers.jacobian={weight: 1.0, kwargs: {target_L: 1.0, n_probes: 1}}"
  [jacobian_fd_dubins]="dubins|50|+loss.regularizers.jacobian={weight: 0.1, kwargs: {target_L: 1.0, n_probes: 1, mode: fd, fd_eps: 0.01}}"
  [jacobian_fd_w1_dubins]="dubins|50|+loss.regularizers.jacobian={weight: 1.0, kwargs: {target_L: 1.0, n_probes: 1, mode: fd, fd_eps: 0.01}}"
  [state_lipschitz_dubins]="dubins|50|+loss.regularizers.state_lipschitz={weight: 0.1, kwargs: {target_L: 1.0}}"
  [state_lipschitz_w1_dubins]="dubins|50|+loss.regularizers.state_lipschitz={weight: 1.0, kwargs: {target_L: 1.0}}"
  [pixel_lipschitz_dubins]="dubins|50|+loss.regularizers.pixel_lipschitz={weight: 1.0, kwargs: {target_L: 0.05, sigma: 0.1}}"
  [jacobian_pusht]="pusht|26|+loss.regularizers.jacobian={weight: 1.0, kwargs: {target_L: 1.0, n_probes: 1}}"
  [pixel_lipschitz_pusht]="pusht|26|+loss.regularizers.pixel_lipschitz={weight: 1.0, kwargs: {target_L: 0.05, sigma: 0.1}}"
)

run_one() {  # preset name
  local name="$1" spec data epochs override
  spec="${RUNS[$name]:-}"
  [ -z "$spec" ] && { echo "[skip] unknown preset: $name" >&2; return 1; }
  IFS='|' read -r data epochs override <<< "$spec"
  local args=(data="$data" "wandb.enabled=${WANDB:-true}" "trainer.max_epochs=${EPOCHS:-$epochs}"
    "output_model_name=$name" "subdir=$name" "hydra.run.dir=outputs/$name")
  [ -n "$override" ] && args+=("$override")
  echo "[$(date +%H:%M:%S)] start $name (data=$data epochs=${EPOCHS:-$epochs})"
  # shellcheck disable=SC2086
  $PY train.py "${args[@]}" $EXTRA_ARGS >"$LOG/lewm_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name"
}

names=("$@"); [ ${#names[@]} -eq 0 ] && names=("${!RUNS[@]}")

# Round-robin presets onto per-GPU serial queues.
IFS=',' read -ra G <<< "$GPUS"; ngpu=${#G[@]}
declare -a Q; for ((g=0; g<ngpu; g++)); do Q[g]=""; done
i=0; for nm in "${names[@]}"; do Q[$((i % ngpu))]+="$nm"$'\n'; i=$((i+1)); done

pids=()
for ((g=0; g<ngpu; g++)); do
  [ -z "${Q[g]}" ] && continue
  ( export CUDA_VISIBLE_DEVICES="${G[g]}"
    while IFS= read -r nm; do [ -n "$nm" ] && run_one "$nm"; done <<< "${Q[g]}"
  ) & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "LEWM RUNS DONE ($fail stream failure(s)); logs: $LOG/lewm_<name>.log"
