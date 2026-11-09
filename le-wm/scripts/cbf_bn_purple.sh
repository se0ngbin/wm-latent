#!/usr/bin/env bash
# Dreamer-framework CBF on purple obstacles (2026-11-09): jac (jacobian_w1_dubins) with its RED-trained margin head and
# reachability critic, world model unadapted vs label-free test-time BN-adapted on 500 safe purple frames
# (bn_adapt_save.py -> jacobian_w1_dubins_bnpurple_safe500). Nothing is retrained. Same collect_trajs eval as
# latent_cbf/scripts/cbf_margin_heads.sh (diffusion policy, CBF filter, 1000 trajectories, same seed per run).
# Heads: gpfail (8000 uniform red states), gpimag (+ imagined latents). Logs run_logs/lewm_cbf_heads/eval_<tag>_<wm>_purple_<variant>.log
set -uo pipefail
L=/home/seongbin/latent; CPY=$L/latent_cbf/.venv/bin/python; ROOT=$L/latent_cbf/src/latent_cbf
CK=/data/seongbin/lewm/checkpoints; OUT=/data/seongbin/dreamer/lewm; LOG=$L/run_logs/lewm_cbf_heads
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH=$L/le-wm DUBINS_OBST_COLOR=purple
W=jacobian_w1_dubins; G=${G:-1}; N=${N:-1000}
declare -A HEAD=([gpfail]=margin_gp_fail_red_s0.pt [gpimag]=margin_gp_imag_red_s0.pt)
declare -A WMCK=([unadapted]=$CK/$W/weights_epoch_50.pt [bnsafe500]=$CK/${W}_bnpurple_safe500/weights_epoch_50.pt)
cd "$ROOT"
for tag in gpfail gpimag; do for v in unadapted bnsafe500; do
  (
    pol=$OUT/$W/seed0_$tag/PyHJ/gp/epoch_id_9/policy.pth
    CUDA_VISIBLE_DEVICES=$G $CPY scripts/collect_trajs.py --controller diffusion --config diffusion_wm --use_wm_prediction \
      --wm_history_length 8 --wm_backend lewm --lewm_run_name $W --lewm_ckpt_path ${WMCK[$v]} --lewm_margin_ckpt $OUT/$W/${HEAD[$tag]} \
      --filter_directory_gp "$pol" --filter_directory_nogp "$pol" --filter_mode cbf \
      --filename "lewm_cbf_gp_${tag}_${W}_seed0_purple_$v" --n_trajectories $N > "$LOG/eval_${tag}_${W}_purple_$v.log" 2>&1
    echo "[$(date +%H:%M)] $tag $v | $(grep -E '^(Success|Collision) rate' "$LOG/eval_${tag}_${W}_purple_$v.log" | tr '\n' ' ')"
  ) &
done; done
wait; echo CBF_BN_PURPLE_DONE
