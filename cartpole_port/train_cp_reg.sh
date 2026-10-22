#!/bin/bash
# Cartpole 3x3 grid cell: base {sigreg, jac, jac+pull} x safety reg {L∞ safety_adv, λ batch_detached}.
# Usage: train_cp_reg.sh NAME GPU BASE(sigreg|jac|jacpull) REG(linf|linf_g0|lam) [WAIT_PID]
# Same snapshots and recipes as the Dubins/SG runs: L∞ -> cfa5e52 (reseed_queue.sh), λ -> bf16b3e (bdet_queue.sh).
name=$1; gpu=$2; base=$3; reg=$4; wait_pid=${5:-}
[ -n "$wait_pid" ] && while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
# GPUs are shared with other chats' runs: wait for >=30 GB free, holding a per-GPU lock until this run has
# allocated (5 min) so queued launches don't all see the same free memory and OOM together.
exec 9>/data/seongbin/lewm/.cp_gpu$gpu.lock; flock 9
until [ "$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 30000 ]; do sleep 60; done
( sleep 300; flock -u 9 ) &
JAC=("+loss.regularizers.jacobian.weight=1.0" "+loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}")
PULL=("+loss.regularizers.action_separation.weight=0.1" "+loss.regularizers.action_separation.kwargs={pull_target:1.0}")
case $base in sigreg) B=();; jac) B=("${JAC[@]}");; jacpull) B=("${JAC[@]}" "${PULL[@]}");; esac
SA='min_steps:2000,ramp_steps:10000,n_sub:32,pgd_steps:2,attack:linf,eps_pix:0.0313725'
case $reg in
  linf)    D=/data/seongbin/lewm/code_safeadv_cfa5e52; K="{$SA,auc_gate:0.95}";;
  linf_g0) D=/data/seongbin/lewm/code_safeadv_cfa5e52; K="{$SA,auc_gate:0.0}";;
  lam)     D=/data/seongbin/lewm/code_safeadv_bf16b3e; K="{$SA,auc_gate:0.0,target:lambda,lambda_mode:batch_detached}";;
esac
mkdir -p $D/.run_$name && cd $D/.run_$name
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled
/home/seongbin/latent/le-wm/.venv/bin/python ../train.py data=cartpole_safety trainer.max_epochs=50 seed=3072 \
  output_model_name=$name subdir=$name wandb.enabled=false "${B[@]}" \
  "+loss.regularizers.safety_adv.weight=1.0" "+loss.regularizers.safety_adv.kwargs=$K" \
  > /home/seongbin/latent/run_logs/cartpole/train_$name.log 2>&1
