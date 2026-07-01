#!/usr/bin/env bash
# latent_cbf experiment driver for the encoder-Lipschitz study. Stages:
#   wm       train dreamer world model(s)        (was enc_lip_sweep[_w1], enc_lip_fd_followup)
#   cbf      PyHJ gp+nogp train + CBF eval per WM (was *_cbf_pipeline, *_multiseed, pixlip, invariance)
#   eval     eval only; MODE=nominal|cbf          (was enc_lip_eval_nominal, enc_lip_cbf_eval)
#   margin   margin-smoothness stats per WM       (was enc_lip_margin)
# (The upstream scripts hp_ablation.sh and collect_batch_diffusion.sh are kept as-is.)
#
# Usage (runs from the repo regardless of cwd):
#   scripts/run.sh wm baseline obs_state image image_fd
#   WEIGHT=1.0 SWEEP=enc_lip_sweep_w1 scripts/run.sh wm obs_state image image_fd
#   SEEDS="0 1 2" scripts/run.sh cbf baseline obs_state image image_fd        # dreamer backend
#   BACKEND=lewm SEEDS="0 1 2" scripts/run.sh cbf no_reg_dubins jacobian_w1_dubins
#   MODE=nominal scripts/run.sh eval baseline obs_state image image_fd
#
# Policies for every WM live at <ckpt-base>/<wm>/seed<seed>/PyHJ/{gp,nogp}/epoch_id_*;
# the latest epoch is selected automatically. Env: GPUS ("0,1"), N_TRAJ (1000),
# SEEDS ("0"), STEPS (40000, wm), WEIGHT/TARGET_L (0.1/1.0, wm), SWEEP (dreamer
# ckpt dir), BACKEND (dreamer|lewm, cbf), MODE (nominal|cbf, eval).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$ROOT"
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
export WANDB_MODE=disabled
: "${GPUS:=0,1}"; : "${N_TRAJ:=1000}"; : "${SEEDS:=0}"
: "${SWEEP:=enc_lip_sweep}"
RUNLOG=/home/seongbin/latent/run_logs
# REPO_ROOT/data is the symlinked machine data root (see configs/paths.py); ROOT is src/latent_cbf.
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
DCK="$REPO_ROOT/data/dreamer/$SWEEP"                  # dreamer ckpt base
LEWM_CK=/data/seongbin/lewm/checkpoints              # lewm WM checkpoints
LEWM_OUT=/data/seongbin/dreamer/lewm                 # lewm margin/policy outputs
BUF=/data/seongbin/dreamer/buffers/dreamer_buffer.h5

toppol(){ ls -1d "$1"/epoch_id_*/ 2>/dev/null | sed -E 's#.*epoch_id_([0-9]+)/#\1 &#' | sort -n | tail -1 | awk '{print $2}'; }
wmckpt(){ ls -1 "$LEWM_CK/$1"/weights_epoch_*.pt 2>/dev/null | sed -E 's/.*epoch_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}'; }

# dispatch FN JOB...  -> round-robin jobs onto per-GPU serial queues, call FN per job.
dispatch(){
  local fn="$1"; shift
  IFS=',' read -ra G <<< "$GPUS"; local ngpu=${#G[@]} g
  declare -a Q; for ((g=0; g<ngpu; g++)); do Q[g]=""; done
  local i=0 j; for j in "$@"; do Q[$((i % ngpu))]+="$j"$'\n'; i=$((i+1)); done
  local pids=()
  for ((g=0; g<ngpu; g++)); do
    [ -z "${Q[g]}" ] && continue
    ( export CUDA_VISIBLE_DEVICES="${G[g]}"
      while IFS= read -r j; do [ -n "$j" ] && "$fn" "$j"; done <<< "${Q[g]}"
    ) & pids+=($!)
  done
  local p; for p in "${pids[@]}"; do wait "$p"; done
}

# ---- stage: wm (dreamer_offline.py) ----
declare -A WMARGS=(
  [baseline]="--enc_lip_weight 0.0"
  [obs_state]="--enc_lip_keys obs_state --enc_lip_probes 4"
  [image]="--enc_lip_keys image --enc_lip_mode exact --enc_lip_probes 1"
  [image_fd]="--enc_lip_keys image --enc_lip_mode fd --enc_lip_fd_eps 0.01 --enc_lip_probes 1"
  [invariance]="--enc_lip_keys image --enc_lip_mode invariance --enc_lip_sigma 0.1 --enc_lip_probes 1"
)
wm_one(){  # wm name
  local name="$1" reg="${WMARGS[$name]:-}"
  [ -z "$reg" ] && { echo "[skip] unknown wm preset: $name" >&2; return 1; }
  [ "$name" = baseline ] || reg="--enc_lip_weight ${WEIGHT:-0.1} --enc_lip_target_L ${TARGET_L:-1.0} $reg"
  mkdir -p "$RUNLOG/$SWEEP"
  echo "[$(date +%H:%M:%S)] WM start $name -> $DCK/$name"
  # shellcheck disable=SC2086
  $PY scripts/dreamer_offline.py --steps "${STEPS:-40000}" --logdir "$DCK/$name" $reg \
    2>&1 | tee "$RUNLOG/$SWEEP/${name}.log"
  echo "[$(date +%H:%M:%S)] WM done  $name"
}

# ---- stage: cbf (PyHJ train gp+nogp, then CBF-filtered eval) ----
cbf_one(){  # "wm:seed"
  local tok="$1" wm seed; IFS=: read -r wm seed <<< "$tok"; seed="${seed:-0}"
  local mode flag gp nogp
  if [ "${BACKEND:-dreamer}" = lewm ]; then
    export PYTHONPATH=/home/seongbin/latent/le-wm   # lewm backend imports jepa/module
    local ckpt margin sd log; ckpt="$(wmckpt "$wm")"
    [ -z "$ckpt" ] && { echo "[skip] no lewm ckpt for $wm" >&2; return; }
    margin="$LEWM_OUT/$wm/margin_heads.pt"; sd="$LEWM_OUT/$wm/seed$seed"; log="$RUNLOG/lewm_cbf"; mkdir -p "$log"
    if [ ! -f "$margin" ]; then
      echo "[$(date +%H:%M:%S)] $wm: train margin"
      $PY scripts/train_margin_lewm.py --lewm_run_name "$wm" --lewm_ckpt_path "$ckpt" \
        --buffer_path "$BUF" --out_path "$margin" --steps "${MARGIN_STEPS:-40000}" >"$log/margin_${wm}.log" 2>&1
    fi
    local be=(--wm_backend lewm --lewm_run_name "$wm" --lewm_ckpt_path "$ckpt" --lewm_margin_ckpt "$margin")
    for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
      $PY scripts/wm_ddpg.py "${be[@]}" --logdir "$sd" --seed "$seed" $flag >"$log/train_${wm}_s${seed}_${mode}.log" 2>&1
    done
    gp="$(toppol "$sd/PyHJ/gp")policy.pth"; nogp="$(toppol "$sd/PyHJ/nogp")policy.pth"
    for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
      $PY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction --wm_history_length 8 \
        "${be[@]}" --filter_directory_gp "$gp" --filter_directory_nogp "$nogp" \
        --filter_mode cbf $flag --filename "lewm_cbf_${mode}_${wm}_seed${seed}" --n_trajectories "$N_TRAJ" \
        >"$log/eval_${wm}_s${seed}_${mode}.log" 2>&1
    done
  else
    local rssm sd log; rssm="$DCK/$wm/rssm_ckpt.pt"; sd="$DCK/$wm/seed$seed"; log="$RUNLOG/enc_lip_cbf"; mkdir -p "$log"
    for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
      $PY scripts/wm_ddpg.py --wm_backend dreamer --logdir "$sd" --rssm-ckpt "$rssm" --seed "$seed" $flag \
        >"$log/train_${wm}_s${seed}_${mode}.log" 2>&1
    done
    gp="$(toppol "$sd/PyHJ/gp")policy.pth"; nogp="$(toppol "$sd/PyHJ/nogp")policy.pth"
    for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
      $PY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction --wm_history_length 8 \
        --wm_checkpoint "$rssm" --filter_directory_gp "$gp" --filter_directory_nogp "$nogp" \
        --filter_mode cbf $flag --filename "cbf_${mode}_${wm}_seed${seed}" --n_trajectories "$N_TRAJ" \
        >"$log/eval_${wm}_s${seed}_${mode}.log" 2>&1
    done
  fi
  echo "[$(date +%H:%M:%S)] CBF done $wm seed$seed"
}

# ---- stage: eval (dreamer; no PyHJ training) ----
eval_one(){  # wm name; reads seed ${SEED:-0} for cbf mode
  local wm="$1" seed="${SEED:-0}" log="$RUNLOG/enc_lip_eval"; mkdir -p "$log"
  local rssm="$DCK/$wm/rssm_ckpt.pt"
  if [ "${MODE:-nominal}" = nominal ]; then
    echo "[$(date +%H:%M:%S)] eval nominal $wm"
    $PY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction --wm_history_length 8 \
      --wm_checkpoint "$rssm" --filename "nom_${wm}" --n_trajectories "$N_TRAJ" --filter_mode none \
      2>&1 | tee "$log/nom_${wm}.log" | grep -iE "Success rate|Collision rate" || true
  else
    local sd="$DCK/$wm/seed$seed" gp nogp mode flag
    gp="$(toppol "$sd/PyHJ/gp")policy.pth"; nogp="$(toppol "$sd/PyHJ/nogp")policy.pth"
    if [ ! -f "$gp" ] || [ ! -f "$nogp" ]; then echo "[skip] $wm (no policy under $sd)" >&2; return; fi
    for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
      echo "[$(date +%H:%M:%S)] eval cbf $wm $mode"
      $PY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction --wm_history_length 8 \
        --wm_checkpoint "$rssm" --filter_directory_gp "$gp" --filter_directory_nogp "$nogp" \
        --filter_mode cbf $flag --filename "cbf_${mode}_${wm}" --n_trajectories "$N_TRAJ" \
        2>&1 | tee "$log/cbf_${mode}_${wm}.log" | grep -iE "Success rate|Collision rate" || true
    done
  fi
}

# ---- stage: margin (smoothness stats) ----
margin_one(){  # wm name
  local wm="$1" seed="${SEED:-0}" log="$RUNLOG/enc_lip_margin"; mkdir -p "$log"
  local rssm="$DCK/$wm/rssm_ckpt.pt" sd="$DCK/$wm/seed$seed"
  $PY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction --wm_history_length 8 \
    --wm_checkpoint "$rssm" --filter_directory_gp "$(toppol "$sd/PyHJ/gp")policy.pth" \
    --filter_directory_nogp "$(toppol "$sd/PyHJ/nogp")policy.pth" \
    --filter_mode cbf --filename "margin_$wm" --n_trajectories "${N_TRAJ:-50}" --save_images >"$log/eval_$wm.log" 2>&1
  $PY scripts/wm_trajectory_stats.py --rssm-ckpt "$rssm" \
    --traj-h5 /data/seongbin/dreamer/trajs/margin_$wm.h5 --hist 5 >"$log/stats_$wm.log" 2>&1
  echo "[$(date +%H:%M:%S)] margin $wm: $(grep -E 'max_diffl_gp|margin_gp\| mean' "$log/stats_$wm.log" | tr '\n' ' ')"
}

eval_summary(){  # wm...   nominal-eval table
  echo "===== EVAL SUMMARY (success | collision) ====="
  local n f sr cr
  for n in "$@"; do
    f="$RUNLOG/enc_lip_eval/nom_${n}.log"; [ -f "$f" ] || continue
    sr=$(grep -iE "^Success rate:" "$f" | tail -1); cr=$(grep -iE "^Collision rate:" "$f" | tail -1)
    printf "%-12s %s | %s\n" "$n" "${sr:-n/a}" "${cr:-n/a}"
  done
}

stage="${1:-}"; shift 2>/dev/null || true
case "$stage" in
  wm)   [ $# -eq 0 ] && set -- baseline obs_state image image_fd
        dispatch wm_one "$@"; echo "WM DONE ($SWEEP)";;
  cbf)  [ $# -eq 0 ] && { echo "usage: run.sh cbf <wm>..." >&2; exit 1; }
        jobs=(); for wm in "$@"; do for s in $SEEDS; do jobs+=("$wm:$s"); done; done
        dispatch cbf_one "${jobs[@]}"; echo "CBF DONE";;
  eval) [ $# -eq 0 ] && set -- baseline obs_state image image_fd
        dispatch eval_one "$@"; [ "${MODE:-nominal}" = nominal ] && eval_summary "$@"; echo "EVAL DONE";;
  margin) [ $# -eq 0 ] && set -- baseline obs_state image image_fd
        dispatch margin_one "$@"; echo "MARGIN DONE";;
  *) echo "usage: run.sh {wm|cbf|eval|margin} [names...]" >&2; exit 1;;
esac
