#!/usr/bin/env bash
# Diffusion policy + CBF filter under appearance shift — EXACT non-OOD setup
# (run.sh cbf branch, seed0, filter_mode cbf, N=1000); only the obstacle appearance
# changes via the env-var hooks in dubins/dubins_env.py (defaults = red circle,
# byte-identical to the recorded non-OOD run). Models: baseline/jacobian/jacpull
# (lewm backend, gp+nogp) + dreamer baseline (dreamer backend, gp only — no nogp
# critic was trained). Conditions: red (control) / purple (color) / diamond (shape,
# genuine L1 geometry). 4-wide: 2 jobs per GPU (0,1).
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
LOG=/home/seongbin/latent/run_logs/diffusion_cbf_ood; mkdir -p "$LOG"
N_TRAJ=${N_TRAJ:-1000}
LEWM_CK=/data/seongbin/lewm/checkpoints; LEWM_OUT=/data/seongbin/dreamer/lewm
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
DREAMER_GP=/data/seongbin/dreamer/lewm/_value_ood/dreamer_baseline/PyHJ/gp/epoch_id_9/policy.pth
declare -A LEWM_WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
latest(){ ls -1d "$1"/epoch_id_* 2>/dev/null | sed -E 's#.*epoch_id_([0-9]+)$#\1 &#' | sort -n | tail -1 | awk '{print $2}'; }

run_one(){  # "model:mode:cond:gpu"
  local tok="$1" model mode cond gpu; IFS=: read -r model mode cond gpu <<< "$tok"
  local ev=(); case "$cond" in
    purple)  ev=(DUBINS_OBST_COLOR=purple) ;;
    diamond) ev=(DUBINS_OBST_SHAPE=diamond) ;;
    red)     ev=() ;;
  esac
  local args=(); local flag=""; [ "$mode" = nogp ] && flag="--no_gp"
  if [ "$model" = dreamer ]; then
    args=(--wm_checkpoint "$RSSM" --filter_directory_gp "$DREAMER_GP" --filter_directory_nogp "$DREAMER_GP")
  else
    local wm=${LEWM_WM[$model]}
    local ckpt=$LEWM_CK/$wm/weights_epoch_50.pt margin=$LEWM_OUT/$wm/margin_heads.pt
    local gp="$(latest "$LEWM_OUT/$wm/seed0/PyHJ/gp")/policy.pth"
    local nogp="$(latest "$LEWM_OUT/$wm/seed0/PyHJ/nogp")/policy.pth"
    args=(--wm_backend lewm --lewm_run_name "$wm" --lewm_ckpt_path "$ckpt" --lewm_margin_ckpt "$margin" \
          --filter_directory_gp "$gp" --filter_directory_nogp "$nogp")
  fi
  local out="$LOG/${model}_${cond}_${mode}.log"
  echo "[$(date +%H:%M:%S)] start $model/$cond/$mode gpu$gpu"
  env "${ev[@]}" CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/collect_trajs.py --controller diffusion \
    --config diffusion_wm --use_wm_prediction --wm_history_length 8 "${args[@]}" \
    --filter_mode cbf $flag --filename "diffcbf_${model}_${cond}_${mode}" --n_trajectories "$N_TRAJ" \
    > "$out" 2>&1
  echo "[$(date +%H:%M:%S)] done  $model/$cond/$mode -> $(grep -iE 'Success rate|Collision rate' "$out" | tr '\n' ' ')"
}

# build job list: lewm gp+nogp, dreamer gp; x red/purple/diamond
JOBS=()
for model in baseline jacobian jacpull; do for mode in gp nogp; do for cond in red purple diamond; do
  JOBS+=("$model:$mode:$cond"); done; done; done
for cond in red purple diamond; do JOBS+=("dreamer:gp:$cond"); done

# 4 lanes: lane0,1 -> gpu0 ; lane2,3 -> gpu1
declare -a LANE=("" "" "" ""); GPUOF=(0 0 1 1)
for i in "${!JOBS[@]}"; do l=$((i % 4)); LANE[$l]+="${JOBS[$i]}:${GPUOF[$l]}"$'\n'; done
pids=()
for l in 0 1 2 3; do
  ( while IFS= read -r j; do [ -n "$j" ] && run_one "$j"; done <<< "${LANE[$l]}" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "DIFFUSION_CBF_OOD_DONE"; echo "=== summary (success / collision) ==="
for model in baseline jacobian jacpull dreamer; do for cond in red purple diamond; do for mode in gp nogp; do
  f="$LOG/${model}_${cond}_${mode}.log"; [ -f "$f" ] || continue
  echo "$model/$cond/$mode: $(grep -iE 'Success rate|Collision rate' "$f" | tr '\n' ' ')"
done; done; done