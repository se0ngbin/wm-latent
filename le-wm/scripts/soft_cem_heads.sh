#!/usr/bin/env bash
# Red rk0 s0 HJ margin heads for the safety-reg WMs (same args as ood_margin.sh rk=0).
LCBF=/home/seongbin/latent/latent_cbf; MDIR=/data/seongbin/dreamer/lewm; LOG=/home/seongbin/latent/run_logs
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
for ck in lewm_dubins_safeadv_linf50 lewm_dubins_jacpull_safeadv_linf50; do
  mkdir -p "$MDIR/$ck"
  CUDA_VISIBLE_DEVICES=${1:-0} "$LCBF/.venv/bin/python" -u "$LCBF/src/latent_cbf/scripts/train_margin_hj_lewm.py" \
    --lewm_ckpt "$ck/weights_epoch_50.pt" --out_path "$MDIR/$ck/margin_hj_color_red_rk0_s0.pt" \
    --obstacle_color red --n_states 12000 --rollout_k 0 --steps 4000 --seed 0 \
    > "$LOG/ood_margin_train_color_${ck}_red_rk0_s0.log" 2>&1
  echo "$ck $(grep -E "val_sign_acc" "$LOG/ood_margin_train_color_${ck}_red_rk0_s0.log" | tail -1)"
done
