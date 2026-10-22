#!/bin/bash
# Cartpole safety-direction (λ-projected) reg, batch_detached: identical recipe to the Dubins/SG `lam_bdet` runs
# (safeadv_results/bdet_queue.sh), same snapshot. sigreg base, no AUC gate.
GPU=${GPU:-0}
D=/data/seongbin/lewm/code_safeadv_bf16b3e; name=cp_lam_bdet50
mkdir -p $D/.run_$name && cd $D/.run_$name
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled
/home/seongbin/latent/le-wm/.venv/bin/python ../train.py data=cartpole_safety trainer.max_epochs=50 seed=3072 \
  output_model_name=$name subdir=$name wandb.enabled=false "+loss.regularizers.safety_adv.weight=1.0" \
  "+loss.regularizers.safety_adv.kwargs={min_steps:2000,auc_gate:0.0,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725,target:lambda,lambda_mode:batch_detached}" \
  > /home/seongbin/latent/run_logs/cartpole/train_$name.log 2>&1
