#!/usr/bin/env bash
# Sweep training runs:
#   - 3 alternative regularizers x {sigreg on, sigreg off} = 6 runs
#   - 2 baselines: sigreg_only (original LE-WM), no_reg (pred_loss only)
# => 8 runs total.
#
# Override via env:
#   DATA=pusht (default)         EPOCHS=100 (default)
#   STABLEWM_HOME=/data/seongbin/lewm
#   GPUS="0"                     # comma-sep GPU ids. One run pinned per GPU
#                                # in parallel; serial per-GPU. e.g. "0,1" -> 2x.
#   ONLY="vicreg_sigreg_off"     # comma-separated run names; runs only these
#   EXTRA_ARGS="..."             # appended verbatim to every train.py invocation

set -uo pipefail

# STABLEWM_HOME is the root for both datasets and checkpoints:
#   <STABLEWM_HOME>/datasets/<name>.h5
#   <STABLEWM_HOME>/checkpoints/<run_name>/weights_epoch_*.pt
: "${STABLEWM_HOME:=/data/seongbin/lewm}"
# : "${EPOCHS:=100}"
: "${EPOCHS:=50}"
: "${DATA:=pusht}"
: "${EXTRA_ARGS:=}"
: "${ONLY:=}"
: "${GPUS:=0}"
export STABLEWM_HOME

PY=.venv/bin/python
mkdir -p logs

# Each entry: run_name|flag|other_name|other_hydra_val
# flag: baseline_on | baseline_off | on | off
RUNS=(
  "sigreg_only|baseline_on||"
  "no_reg|baseline_off||"
  # "temporal_lipschitz__sigreg_on|on|temporal_lipschitz|{weight: 1.0, kwargs: {gamma: 1.0}}"
  # "temporal_lipschitz__sigreg_off|off|temporal_lipschitz|{weight: 1.0, kwargs: {gamma: 1.0}}"
  # "vicreg__sigreg_on|on|vicreg|{weight: 1.0, kwargs: {var_weight: 1.0, cov_weight: 0.04, std_target: 1.0}}"
  # "vicreg__sigreg_off|off|vicreg|{weight: 1.0, kwargs: {var_weight: 1.0, cov_weight: 0.04, std_target: 1.0}}"
  # "jacobian__sigreg_on|on|jacobian|{weight: 0.1, kwargs: {target_L: 1.0, n_probes: 1}}"
  # "jacobian__sigreg_off|off|jacobian|{weight: 0.1, kwargs: {target_L: 1.0, n_probes: 1}}"
)

should_run() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}

dispatch_run() {
  local spec="$1"
  local run_name flag name val
  IFS='|' read -r run_name flag name val <<< "$spec"
  local tagged="${run_name}_${DATA}"
  echo
  echo "================================================================"
  echo "[$(date +%H:%M:%S)] [GPU ${CUDA_VISIBLE_DEVICES:-?}] Run: $tagged"
  echo "================================================================"
  local args=(
    data="$DATA"
    wandb.enabled=false
    trainer.max_epochs="$EPOCHS"
    "output_model_name=$tagged"
    "subdir=$tagged"
    "hydra.run.dir=outputs/$tagged"
  )
  case "$flag" in
    baseline_off) args+=('~loss.regularizers.sigreg') ;;
    on)           args+=("+loss.regularizers.${name}=${val}") ;;
    off)          args+=("+loss.regularizers.${name}=${val}" '~loss.regularizers.sigreg') ;;
    baseline_on)  : ;;
  esac
  # shellcheck disable=SC2086
  $PY train.py "${args[@]}" $EXTRA_ARGS 2>&1 | tee "logs/${tagged}.log"
}

# Filter by ONLY and round-robin assign each run to a GPU queue.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPUS=${#GPU_ARR[@]}
declare -a QUEUE
for ((g=0; g<NGPUS; g++)); do QUEUE[$g]=""; done

idx=0
for spec in "${RUNS[@]}"; do
  run_name="${spec%%|*}"
  if ! should_run "$run_name"; then
    echo "[skip] $run_name (not in ONLY=$ONLY)"
    continue
  fi
  g=$(( idx % NGPUS ))
  # Newline-separated per-GPU queue.
  QUEUE[$g]+="$spec"$'\n'
  idx=$((idx+1))
done

# Launch one subshell per GPU; each processes its queue serially.
pids=()
for ((g=0; g<NGPUS; g++)); do
  gpu="${GPU_ARR[$g]}"
  queue="${QUEUE[$g]}"
  [ -z "$queue" ] && continue
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    while IFS= read -r spec; do
      [ -z "$spec" ] && continue
      dispatch_run "$spec"
    done <<< "$queue"
  ) &
  pids+=($!)
done

# Wait, track failures.
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail+1))
done

echo
if [ $fail -gt 0 ]; then
  echo "WARNING: $fail GPU stream(s) had failures — check logs/*.log"
fi
echo "Checkpoints under: $STABLEWM_HOME/checkpoints/<run_name>/"
echo "Logs: logs/*.log"
