#!/bin/bash
# Save layout-disjoint margin heads (mlp + linear, eval seed 0) for the Point CEM models.
cd "$(dirname "$0")"
D=/data/seongbin/lewm/datasets
CUDA_VISIBLE_DEVICES=${GPU:-1} STABLEWM_HOME=/data/seongbin/lewm HEAD=both SG_SEED=0 SAVE_HEADS=/data/seongbin/lewm/sg_point_heads \
  TRAIN_H5=$D/sg_point_ovh224.h5 SG_DATA=$D/sg_point_ovh224_multi_eval.npz \
  DIAG_ENCODERS=${ENC:-pt_baseline_ovh,pt_jacobian_ovh,pt_jac_safeadv_linf_ovh,pt_jacpull_safeadv_linf_ovh} \
  /home/seongbin/latent/le-wm/.venv/bin/python sg_point_bigprobe.py
