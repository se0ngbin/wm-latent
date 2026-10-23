#!/usr/bin/env bash
# Supervise the 9 clean-Point WMs, then eval. Every 5 min: a run that was launched (a "train NAME" line in any launcher
# log) but has no epoch-50 checkpoint, no live train.py and no pending retry is DEAD (mid-run OOM: the GPUs are shared with
# other jobs) -> relaunch it from scratch via run_point.sh NAME (memory-aware), at most 3 retries. When all 9 checkpoints
# and the multi-variant eval set exist, run sg_point_eval.py for eval seeds 0-2 and print the SGPT lines.
cd "$(dirname "${BASH_SOURCE[0]}")"
CODE=/data/seongbin/lewm/code_point_92e7589; CK=/data/seongbin/lewm/checkpoints; L=/home/seongbin/latent/run_logs/sg_point
LOGS="/home/seongbin/latent/run_logs/sg_point_launcher.log /home/seongbin/latent/run_logs/sg_point_queue.log $L/retry.log"
NAMES="pt_baseline pt_jacobian pt_jacpull pt_safeadv_linf_g0 pt_jac_safeadv_linf pt_jacpull_safeadv_linf pt_lam_bdet pt_jac_lam_bdet pt_jacpull_lam_bdet"
declare -A TRIES
while :; do
  missing=0
  for n in $NAMES; do
    [ -f "$CK/$n/weights_epoch_50.pt" ] && continue
    missing=1
    grep -qs "train $n " $LOGS || continue                         # not launched yet (still in the main queue)
    pgrep -f "output_model_name=$n " > /dev/null && continue        # training
    pgrep -f "run_point.sh $n\$" > /dev/null && continue            # retry waiting for GPU memory
    TRIES[$n]=$(( ${TRIES[$n]:-0} + 1 ))
    [ "${TRIES[$n]}" -gt 3 ] && { echo "GAVE UP on $n after 3 retries"; tail -5 "$L/train_$n.log"; exit 1; }
    echo "[$(date +%H:%M)] $n died without a checkpoint -> retry ${TRIES[$n]}"
    rm -rf "$CODE/.run_$n"; nohup "$CODE/run_point.sh" "$n" >> "$L/retry.log" 2>&1 &
  done
  [ $missing = 0 ] && [ -f /data/seongbin/lewm/datasets/sg_point_multi_eval.npz ] && break
  sleep 300
done
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$gpu SG_SEED=$s /home/seongbin/latent/le-wm/.venv/bin/python sg_point_eval.py > $L/eval_seed$s.log 2>&1; done
grep -h SGPT $L/eval_seed*.log
