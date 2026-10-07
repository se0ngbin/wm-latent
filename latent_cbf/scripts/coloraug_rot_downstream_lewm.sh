#!/usr/bin/env bash
# LEWM slice of coloraug_downstream (run when the LEWM color-aug trio finishes).
#   A. Margin OOD AUC (probe margin, 3 axes; ood_margin_gp_coloraug_rot.py has the coloraug ENCODERS)
#   B. Value OOD AUC: margin heads (red buffer) -> gp critic -> critic_ood_eval, per encoder.
set -uo pipefail
LEWM=/home/seongbin/latent/le-wm; LCBF=/home/seongbin/latent/latent_cbf
CK=/data/seongbin/lewm/checkpoints; MGD=/data/seongbin/dreamer/lewm
BUF=/data/seongbin/dreamer/buffers/dreamer_buffer.h5
LOG=/home/seongbin/latent/run_logs/coloraug_rot; mkdir -p "$LOG"
LPY=$LEWM/.venv/bin/python; CPY=$LCBF/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
declare -A WM=( [baseline]=coloraug_rot_baseline [jacobian]=coloraug_rot_jacobian [jacpull]=coloraug_rot_jacpull )

echo "[$(date +%H:%M:%S)] A. LEWM margin OOD AUC (probe)"
[ "${SKIP_A:-0}" = 1 ] || for axis in color shape rotate; do
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=0 "$LPY" -u scripts/ood_margin_gp_coloraug_rot.py "$axis" 0 ) \
    > "$LOG/margin_coloraug_jepa_${axis}.log" 2>&1
  grep "KEY jepa_gp" "$LOG/margin_coloraug_jepa_${axis}.log"
done

cd "$LCBF/src/latent_cbf"; export PYTHONPATH=/home/seongbin/latent/le-wm
lewm_value() {  # tag gpu
  local tag=$1 gpu=$2; local w=${WM[$tag]}
  local ckpt=$CK/$w/weights_epoch_50.pt margin=$MGD/$w/margin_heads.pt logdir=$MGD/$w/seed0
  mkdir -p "$MGD/$w"
  [ -f "$margin" ] || { echo "[$(date +%H:%M:%S)] margin $tag";
    CUDA_VISIBLE_DEVICES=$gpu "$CPY" scripts/train_margin_lewm.py --lewm_run_name "$w" \
      --lewm_ckpt_path "$ckpt" --buffer_path "$BUF" --out_path "$margin" --steps 40000 \
      > "$LOG/marginheads_$tag.log" 2>&1 || { echo " MARGIN $tag FAIL"; return 1; }; }
  local pol; pol=$(ls -d "$logdir"/PyHJ/gp/epoch_id_* 2>/dev/null | head -1)
  [ -n "$pol" ] && [ -f "$pol/policy.pth" ] || { echo "[$(date +%H:%M:%S)] critic $tag";
    CUDA_VISIBLE_DEVICES=$gpu "$CPY" scripts/wm_ddpg.py --wm_backend lewm --lewm_run_name "$w" \
      --lewm_ckpt_path "$ckpt" --lewm_margin_ckpt "$margin" --logdir "$logdir" --seed 0 \
      > "$LOG/critic_$tag.log" 2>&1 || { echo " CRITIC $tag FAIL"; return 1; }; }
  pol=$(ls -d "$logdir"/PyHJ/gp/epoch_id_* | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth
  CUDA_VISIBLE_DEVICES=$gpu "$CPY" scripts/critic_ood_eval.py --tag "coloraug_rot_$tag" \
    --lewm_ckpt "$ckpt" --margin_ckpt "$margin" --policy "$pol" > "$LOG/value_$tag.log" 2>&1
  grep "^KEY critic" "$LOG/value_$tag.log"
}
echo "[$(date +%H:%M:%S)] B. LEWM value (baseline+jacobian ‖, then jacpull)"
lewm_value baseline 0 & lewm_value jacobian 1 & wait
lewm_value jacpull 0
echo "COLORAUG_ROT_LEWM_DOWNSTREAM_DONE"
