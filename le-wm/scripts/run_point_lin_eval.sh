#!/bin/bash
# Linear vs MLP margin function on overhead@224 Point latents (layout-disjoint, correct orientation), all finished WMs x eval seeds 0-2.
cd "$(dirname "$0")"
D=/data/seongbin/lewm/datasets; L=/home/seongbin/latent/run_logs/sg_point
ENC=$(for d in /data/seongbin/lewm/checkpoints/pt_*_ovh*; do [ -f $d/weights_epoch_50.pt ] && echo -n "$(basename $d),"; done); ENC=${ENC%,}
for s in 0 1 2; do gpu=$([ $s = 0 ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$gpu STABLEWM_HOME=/data/seongbin/lewm HEAD=both SG_SEED=$s TRAIN_H5=$D/sg_point_ovh224.h5 SG_DATA=$D/sg_point_ovh224_multi_eval.npz DIAG_ENCODERS=$ENC \
    nohup /home/seongbin/latent/le-wm/.venv/bin/python sg_point_bigprobe.py > $L/bigprobe_both_ovh_seed$s.log 2>&1 &
done
