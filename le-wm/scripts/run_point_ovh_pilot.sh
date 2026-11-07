#!/usr/bin/env bash
# Overhead@224 pilot: wait for the merged dataset, then train baseline + jac+pull (seed 3072, suffix _ovh) on it.
cd "$(dirname "${BASH_SOURCE[0]}")"
until [ -f /data/seongbin/lewm/datasets/sg_point_ovh224.h5 ] && ! pgrep -f "merge_shards.py .*point_ovh224" > /dev/null; do sleep 60; done
DATA=sg_point_ovh224_safety SFX=_ovh ./run_point.sh pt_baseline pt_jacpull
