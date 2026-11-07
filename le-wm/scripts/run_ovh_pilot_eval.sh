#!/usr/bin/env bash
# Pilot verdict: layout-disjoint margin probe (sg_point_bigprobe.py) for the overhead@224 WMs vs the fixedfar@128 ones.
cd "$(dirname "${BASH_SOURCE[0]}")"
CK=/data/seongbin/lewm/checkpoints; D=/data/seongbin/lewm/datasets; L=/home/seongbin/latent/run_logs/sg_point
until [ -f $CK/pt_baseline_ovh/weights_epoch_50.pt ] && [ -f $CK/pt_jacpull_ovh/weights_epoch_50.pt ] && [ -f $D/sg_point_ovh224_multi_eval.npz ]; do
  grep -q FAILED /home/seongbin/latent/run_logs/sg_point_ovh_pilot.log && { echo "pilot training FAILED:"; cat /home/seongbin/latent/run_logs/sg_point_ovh_pilot.log; exit 1; }
  sleep 300
done
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
PY=/home/seongbin/latent/le-wm/.venv/bin/python
CUDA_VISIBLE_DEVICES=$gpu TRAIN_H5=$D/sg_point_ovh224.h5 SG_DATA=$D/sg_point_ovh224_multi_eval.npz DIAG_ENCODERS=pt_baseline_ovh,pt_jacpull_ovh $PY sg_point_bigprobe.py > $L/bigprobe_ovh.log 2>&1
CUDA_VISIBLE_DEVICES=$gpu DIAG_ENCODERS=pt_baseline,pt_jacpull $PY sg_point_bigprobe.py > $L/bigprobe_fixedfar_refit.log 2>&1
grep -h "^BIG" $L/bigprobe_ovh.log $L/bigprobe_fixedfar_refit.log
