#!/usr/bin/env bash
# Overhead@224 version of sg_point_clean.h5: SAME seeds (20-23) and collector args, so trajectories/layouts are identical
# (rendering does not consume the rng) and only the camera differs. Then merge.
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL=egl
PYV=/home/seongbin/latent/safety-gymnasium/.venv/bin/python
S=/data/seongbin/lewm/datasets/sg_shards; L=/home/seongbin/latent/run_logs/sg_point
for i in 0 1 2 3; do
  $PYV collect_dataset.py --out $S/point_ovh224_s$i.h5 --frames 42500 --agent Point --clean --ep_steps 200 --p_loiter 0.6 \
    --seed $((20 + i)) --cam overhead --size 224 > $L/collect_ovh224_s$i.log 2>&1 &
done
wait
/home/seongbin/latent/le-wm/.venv/bin/python merge_shards.py "$S/point_ovh224_s*.h5" /data/seongbin/lewm/datasets/sg_point_ovh224.h5
