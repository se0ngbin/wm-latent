#!/usr/bin/env bash
# Downstream eval for the obstacle color-aug WMs (Post Sept 16). Stage 1 = WM-only aug;
# downstream trained by the ORIGINAL procedure so the WM change is isolated.
#   A. Margin OOD AUC (probe margin trained fresh per encoder): red/purple/diamond/rot90.
#   B. Value OOD AUC: train margin heads (LEWM, red buffer) -> gp critic -> critic_ood_eval.
# Dreamer's margin is jointly color-aug'd in the RSSM ckpt (caveat, noted in report).
# Compares purple AUC coloraug-vs-original. Run AFTER train_coloraug*.sh finish.
set -uo pipefail
LEWM=/home/seongbin/latent/le-wm; LCBF=/home/seongbin/latent/latent_cbf
CK=/data/seongbin/lewm/checkpoints
MGD=/data/seongbin/dreamer/lewm; DR=/data/seongbin/dreamer/dreamer
BUF=/data/seongbin/dreamer/buffers/dreamer_buffer.h5
RSSM=$DR/enc_lip_sweep/coloraug_baseline/rssm_ckpt.pt
LOG=/home/seongbin/latent/run_logs/coloraug; mkdir -p "$LOG"
LPY=$LEWM/.venv/bin/python; CPY=$LCBF/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled
declare -A WM=( [baseline]=coloraug_baseline [jacobian]=coloraug_jacobian [jacpull]=coloraug_jacpull )

# ---------- A. Margin OOD AUC (self-contained probe margin) ----------
echo "[$(date +%H:%M:%S)] A. margin OOD AUC"
for axis in color shape rotate; do
  ( cd "$LEWM" && CUDA_VISIBLE_DEVICES=0 "$LPY" -u scripts/ood_margin_gp_coloraug.py "$axis" 0 ) \
    > "$LOG/margin_coloraug_jepa_${axis}.log" 2>&1
  grep "KEY jepa_gp" "$LOG/margin_coloraug_jepa_${axis}.log"
  CUDA_VISIBLE_DEVICES=0 "$CPY" -u "$LCBF/src/latent_cbf/scripts/ood_margin_gp_dreamer.py" \
    --rssm_ckpt "$RSSM" --axis "$axis" --tag coloraug --seed 0 \
    > "$LOG/margin_coloraug_dreamer_${axis}.log" 2>&1
  grep "KEY dreamer" "$LOG/margin_coloraug_dreamer_${axis}.log"
done

# ---------- B. Value OOD AUC ----------
cd "$LCBF/src/latent_cbf"
export PYTHONPATH=/home/seongbin/latent/le-wm
# B1. LEWM margin heads (red buffer) + gp critic
lewm_value() {  # tag gpu
  local tag=$1 gpu=$2 w=${WM[$tag]}
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
  CUDA_VISIBLE_DEVICES=$gpu "$CPY" scripts/critic_ood_eval.py --tag "coloraug_$tag" \
    --lewm_ckpt "$ckpt" --margin_ckpt "$margin" --policy "$pol" > "$LOG/value_$tag.log" 2>&1
  grep "^KEY critic" "$LOG/value_$tag.log"
}
echo "[$(date +%H:%M:%S)] B. LEWM value (baseline+jacobian on 0/1, then jacpull)"
lewm_value baseline 0 & lewm_value jacobian 1 & wait
lewm_value jacpull 0

# B2. Dreamer critic (margin already in ckpt) + value eval
echo "[$(date +%H:%M:%S)] B2. dreamer value"
dlog=$DR/enc_lip_sweep/coloraug_baseline/critic_seed0
dpol=$(ls -d "$dlog"/PyHJ/gp/epoch_id_* 2>/dev/null | head -1)
[ -n "$dpol" ] && [ -f "$dpol/policy.pth" ] || CUDA_VISIBLE_DEVICES=1 "$CPY" scripts/wm_ddpg.py \
  --wm_backend dreamer --rssm-ckpt "$RSSM" --logdir "$dlog" --seed 0 > "$LOG/critic_dreamer.log" 2>&1
dpol=$(ls -d "$dlog"/PyHJ/gp/epoch_id_* | sed -E 's#.*epoch_id_([0-9]+)#\1 &#' | sort -n | tail -1 | awk '{print $2}')/policy.pth
CUDA_VISIBLE_DEVICES=1 "$CPY" scripts/critic_ood_eval_dreamer.py --tag "coloraug_dreamer" \
  --rssm_ckpt "$RSSM" --policy "$dpol" > "$LOG/value_dreamer.log" 2>&1
grep "^KEY critic" "$LOG/value_dreamer.log"
echo "COLORAUG_DOWNSTREAM_DONE"
