#!/usr/bin/env bash
# Supervise the clean-Point WMs, then eval. SFXS = training-seed suffixes to supervise ("" = seed 3072, "_s1" = seed 1, ...).
# Every 5 min: a run that was launched (a "train NAME" line in any launcher log) but has no epoch-50 checkpoint, no live
# train.py and no pending retry is DEAD (mid-run OOM: the GPUs are shared) -> relaunch it from scratch via
# run_point.sh (memory-aware), at most 3 retries. When all checkpoints and the eval set exist, run sg_point_eval.py for
# eval seeds 0-2 per suffix and print the SGPT lines.
# usage: SFXS="_s1 _s2" ./run_point_eval_when_ready.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
CODE=/data/seongbin/lewm/code_point_92e7589; CK=/data/seongbin/lewm/checkpoints; L=/home/seongbin/latent/run_logs/sg_point
LOGS="/home/seongbin/latent/run_logs/sg_point_launcher.log /home/seongbin/latent/run_logs/sg_point_queue.log /home/seongbin/latent/run_logs/sg_point_queue_seeds.log $L/retry.log"
BASES="pt_baseline pt_jacobian pt_jacpull pt_safeadv_linf_g0 pt_jac_safeadv_linf pt_jacpull_safeadv_linf pt_lam_bdet pt_jac_lam_bdet pt_jacpull_lam_bdet"
read -ra SFXL <<< "${SFXS:-}"; [ ${#SFXL[@]} = 0 ] && SFXL=("")   # "" = the seed-3072 runs
seed_of(){ case $1 in "") echo 3072;; _s*) echo "${1#_s}";; esac; }
declare -A TRIES
while :; do
  missing=0
  for sfx in "${SFXL[@]}"; do for b in $BASES; do n=$b$sfx
    [ -f "$CK/$n/weights_epoch_50.pt" ] && continue
    missing=1
    grep -qs "train $n " $LOGS || continue                                         # not launched yet (still queued)
    pgrep -f "output_model_name=$n " > /dev/null && continue                        # training
    [ -f "$L/retry_$n.pid" ] && kill -0 "$(cat "$L/retry_$n.pid")" 2>/dev/null && continue   # retry waiting for memory
    TRIES[$n]=$(( ${TRIES[$n]:-0} + 1 ))
    [ "${TRIES[$n]}" -gt 3 ] && { echo "GAVE UP on $n after 3 retries"; tail -5 "$L/train_$n.log"; exit 1; }
    echo "[$(date +%H:%M)] $n died without a checkpoint -> retry ${TRIES[$n]}"
    rm -rf "$CODE/.run_$n"
    SEED=$(seed_of "$sfx") SFX=$sfx nohup "$CODE/run_point.sh" "$b" >> "$L/retry.log" 2>&1 &
    echo $! > "$L/retry_$n.pid"
  done; done
  [ $missing = 0 ] && [ -f /data/seongbin/lewm/datasets/sg_point_multi_eval.npz ] && break
  sleep 300
done
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
for sfx in "${SFXL[@]}"; do for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$gpu SG_SFX=$sfx SG_SEED=$s /home/seongbin/latent/le-wm/.venv/bin/python sg_point_eval.py > "$L/eval${sfx}_seed$s.log" 2>&1
done; done
for sfx in "${SFXL[@]}"; do grep -h SGPT "$L"/eval${sfx}_seed*.log; done
