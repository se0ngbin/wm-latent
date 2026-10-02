#!/usr/bin/env bash
# Multi-seed OOD margin sweep at rollout_k=0 (static; removes the predictor-OOD
# confound). JEPA (3 encoders, rk=0) + Dreamer (3 variants, inherently static),
# axes {color, rotate}, seeds {0,1,2}. Aggregate with ood_seed_aggregate.py.
# Usage: ood_seed_sweep.sh            (all)  |  ood_seed_sweep.sh <axis> "<seeds>"
set -uo pipefail
AXES=${1:-"color rotate"}
SEEDS=${2:-"0 1 2"}
LEWM=/home/seongbin/latent/le-wm

for axis in $AXES; do
  for s in $SEEDS; do
    echo "=== JEPA $axis rk=0 seed=$s ==="
    bash "$LEWM/scripts/ood_margin.sh" "$axis" 0 "$s"
    echo "=== DREAMER $axis seed=$s ==="
    bash "$LEWM/scripts/ood_margin_dreamer.sh" "$axis" "$s"
  done
done
echo "OOD_SEED_SWEEP_DONE"
