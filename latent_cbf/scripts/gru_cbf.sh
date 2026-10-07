#!/usr/bin/env bash
# Baseline GRU CBF pipeline: margin(704-d, memory) -> gp+nogp critics -> closed-loop
# diffusion+CBF OOD (red/purple/diamond). Tests whether the GRU's non-re-grounding
# (memory accumulates the nuisance) shows up as real collisions under color.
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
GPU=${GPU:-1}; N_TRAJ=${N_TRAJ:-1000}
LOG=/home/seongbin/latent/run_logs/gru_cbf; mkdir -p "$LOG"
wm=lewm_gru_dubins
ckpt=/data/seongbin/lewm/checkpoints/$wm/weights_epoch_50.pt
margin=/data/seongbin/dreamer/lewm/$wm/margin_heads.pt
sd=/data/seongbin/dreamer/lewm/$wm/seed0
BUF=/data/seongbin/dreamer/buffers/dreamer_buffer.h5

echo "[$(date +%H:%M:%S)] 1/3 margin"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_margin_lewm.py --lewm_run_name "$wm" --lewm_ckpt_path "$ckpt" \
  --buffer_path "$BUF" --out_path "$margin" --steps 40000 > "$LOG/margin.log" 2>&1 && echo "  margin done" || { echo "  MARGIN FAILED"; exit 1; }

for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
  echo "[$(date +%H:%M:%S)] 2/3 critic $mode"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/wm_ddpg.py --wm_backend lewm --lewm_run_name "$wm" --lewm_ckpt_path "$ckpt" \
    --lewm_margin_ckpt "$margin" --logdir "$sd" --seed 0 $flag > "$LOG/train_$mode.log" 2>&1 && echo "  critic $mode done" || echo "  CRITIC $mode FAILED"
done
gp="$(ls -1d "$sd/PyHJ/gp"/epoch_id_* | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth"
nogp="$(ls -1d "$sd/PyHJ/nogp"/epoch_id_* | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth"

echo "[$(date +%H:%M:%S)] 3/3 closed-loop OOD eval"
for cond in red purple diamond; do
  ev=(); case $cond in purple) ev=(DUBINS_OBST_COLOR=purple);; diamond) ev=(DUBINS_OBST_SHAPE=diamond);; esac
  for mode in gp nogp; do flag=""; [ "$mode" = nogp ] && flag="--no_gp"
    env "${ev[@]}" CUDA_VISIBLE_DEVICES=$GPU $PY scripts/collect_trajs.py --controller diffusion --config diffusion_wm \
      --use_wm_prediction --wm_history_length 8 --wm_backend lewm --lewm_run_name "$wm" --lewm_ckpt_path "$ckpt" \
      --lewm_margin_ckpt "$margin" --filter_directory_gp "$gp" --filter_directory_nogp "$nogp" \
      --filter_mode cbf $flag --filename "grucbf_${cond}_${mode}" --n_trajectories "$N_TRAJ" > "$LOG/eval_${cond}_${mode}.log" 2>&1
    echo "  $cond/$mode: $(grep -iE 'Success rate|Collision rate' "$LOG/eval_${cond}_${mode}.log" | tr '\n' ' ')"
  done
done
echo "GRU_CBF_DONE"
