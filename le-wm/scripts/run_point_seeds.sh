#!/usr/bin/env bash
# Training seeds 1 and 2 of all 9 clean-Point recipes through one memory-aware queue (suffixes _s1/_s2).
cd "$(dirname "${BASH_SOURCE[0]}")"
SEED=1 SFX=_s1 NOWAIT=1 ./run_point.sh
SEED=2 SFX=_s2 NOWAIT=1 ./run_point.sh
echo "ALL SEED RUNS LAUNCHED"
