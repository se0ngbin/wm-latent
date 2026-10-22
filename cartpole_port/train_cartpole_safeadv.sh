#!/bin/bash
# Cartpole jac+pull + L∞ safety reg: identical recipe to the Dubins `jplinf` runs (safeadv_results/reseed_queue.sh),
# same code snapshot. Safety head trains on `failures` (l < 0, "currently safe" margin).
GPU=${GPU:-1}
D=/data/seongbin/lewm/code_safeadv_cfa5e52; name=cp_jacpull_safeadv_linf50
mkdir -p $D/.run_$name && cd $D/.run_$name
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled
/home/seongbin/latent/le-wm/.venv/bin/python ../train.py data=cartpole_safety trainer.max_epochs=50 \
  output_model_name=$name subdir=$name wandb.enabled=false \
  "+loss.regularizers.jacobian.weight=1.0" "+loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}" \
  "+loss.regularizers.action_separation.weight=0.1" "+loss.regularizers.action_separation.kwargs={pull_target:1.0}" \
  "+loss.regularizers.safety_adv.weight=1.0" \
  "+loss.regularizers.safety_adv.kwargs={min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725,auc_gate:0.95}" \
  > /home/seongbin/latent/run_logs/cartpole/train_$name.log 2>&1
