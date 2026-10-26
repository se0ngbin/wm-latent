#!/usr/bin/env bash
# Benchmark learnability check: same collector (collect_dataset.py, clean Point policy mix) for train and test, different
# seeds -> test layouts unseen. Per config: 4 train shards x 10k frames + 1 test shard x 5k frames, all in parallel.
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL=egl
PYV=/home/seongbin/latent/safety-gymnasium/.venv/bin/python
D=/data/seongbin/lewm/datasets/learnability; L=/home/seongbin/latent/run_logs/sg_point/learnability; mkdir -p "$D" "$L"
C=(--agent Point --clean --ep_steps 200 --p_loiter 0.6)
for cfg in "overhead 128" "fixedfar 224"; do set -- $cfg; cam=$1 size=$2; tag=${cam}${size}
  for i in 0 1 2 3; do $PYV collect_dataset.py --out $D/${tag}_train_s$i.h5 --frames 10000 --cam $cam --size $size --seed $((100 + i)) "${C[@]}" > $L/${tag}_train_s$i.log 2>&1 & done
  $PYV collect_dataset.py --out $D/${tag}_test.h5 --frames 5000 --cam $cam --size $size --seed 200 "${C[@]}" > $L/${tag}_test.log 2>&1 &
done
wait; echo LEARNABILITY_DATA_DONE
