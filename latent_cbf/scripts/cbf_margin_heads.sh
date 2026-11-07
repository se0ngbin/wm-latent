#!/usr/bin/env bash
# Dreamer-framework CBF on LeWM with a chosen saved margin head: wm_ddpg (gp) -> collect_trajs CBF eval (1000 trajs).
# Jobs are "<wm>:<tag>", tag -> head file under /data/seongbin/dreamer/lewm/<wm>/:
#   gpfail = margin_gp_fail_red_s0.pt (8000 uniform states)   gpimag = margin_gp_imag_red_s0.pt (+ imagined latents)
#   lin    = margin_lin_fail_red_s0.pt (linear, uniform states)
# Outputs: <wm>/seed0_<tag>/PyHJ/gp, trajs lewm_cbf_gp_<tag>_<wm>_seed0.h5, logs run_logs/lewm_cbf_heads/.
# Usage: scripts/cbf_margin_heads.sh wm:tag ...   (all jobs run in parallel; each waits for a GPU with NEED MiB free)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/latent_cbf" && pwd)"
L=/home/seongbin/latent; CPY=$L/latent_cbf/.venv/bin/python
CK=/data/seongbin/lewm/checkpoints; OUT=/data/seongbin/dreamer/lewm; LOG=$L/run_logs/lewm_cbf_heads; mkdir -p "$LOG"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH=$L/le-wm
NEED=${NEED:-5000}; LOCK=$LOG/.gpu.lock
declare -A HEAD=([gpfail]=margin_gp_fail_red_s0.pt [gpimag]=margin_gp_imag_red_s0.pt [lin]=margin_lin_fail_red_s0.pt)
pick_gpu(){  # ponytail: free-memory poll under a flock, held 120s so the next pick sees this job's memory
  exec 9>"$LOCK"; flock 9
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | awk -v n="$NEED" -F', ' '$2>=n{print $1; exit}')
    [ -n "$g" ] && break; sleep 60
  done
  ( sleep 120; flock -u 9 ) >/dev/null & echo "$g"; exec 9>&-
}
one(){
  local w=${1%%:*} tag=${1##*:}; local head=$OUT/$w/${HEAD[$tag]} ckpt=$CK/$w/weights_epoch_50.pt sd=$OUT/$w/seed0_$tag
  [ -f "$head" ] || { echo "NO HEAD $w $tag"; return; }
  local be=(--wm_backend lewm --lewm_run_name "$w" --lewm_ckpt_path "$ckpt" --lewm_margin_ckpt "$head")
  cd "$ROOT"
  if [ ! -f "$sd/PyHJ/gp/epoch_id_9/policy.pth" ]; then g=$(pick_gpu); echo "[$(date +%H:%M)] $w $tag ddpg (gpu$g)"
    CUDA_VISIBLE_DEVICES=$g $CPY scripts/wm_ddpg.py "${be[@]}" --logdir "$sd" --seed 0 > "$LOG/ddpg_${tag}_$w.log" 2>&1 \
      || { echo "DDPG FAIL $w $tag"; return; }; fi
  local pol=$sd/PyHJ/gp/epoch_id_9/policy.pth
  g=$(pick_gpu); echo "[$(date +%H:%M)] $w $tag eval (gpu$g)"
  CUDA_VISIBLE_DEVICES=$g $CPY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction \
    --wm_history_length 8 "${be[@]}" --filter_directory_gp "$pol" --filter_directory_nogp "$pol" --filter_mode cbf \
    --filename "lewm_cbf_gp_${tag}_${w}_seed0" --n_trajectories 1000 > "$LOG/eval_${tag}_$w.log" 2>&1 \
    || { echo "EVAL FAIL $w $tag"; return; }
  echo "[$(date +%H:%M)] DONE $w $tag | $(grep -E '^(Success|Collision) rate' "$LOG/eval_${tag}_$w.log" | tr '\n' ' ')"
}
for j in "$@"; do one "$j" & sleep 5; done; wait
echo CBF_HEADS_ALL_DONE
