#!/usr/bin/env bash
# Ablations for "why the safety regs plan worse" (see diag_rollout_safety.py):
#  A. raw CEM on the existing adv_h part-ablations (detached gap / encoder term only / predictor term only)
#  B. value functions trained on IMAGINED latents (rollout_k=8) -> diagnostic + safety-first CEM with them
#  C. diagnostic re-run on all models (adds the lambda-projection goal-fidelity test and the rk8 heads)
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
LCBF=/home/seongbin/latent/latent_cbf; MDIR=/data/seongbin/dreamer/lewm; LOG=/home/seongbin/latent/run_logs
PY=/home/seongbin/latent/le-wm/.venv/bin/python
ANY=$MDIR/sigreg_only_dubins/margin_gp_fail_red_s0.pt   # unused at w=0 (raw CEM)
# A: raw CEM (w=0, C=0) on the adv_h ablations, one lane
{ for t in sa_det sa_enc sa_pred; do ./soft_cem.sh 1 lewm_dubins_${t}50 $ANY $t 0 red; done; } &
# B1: imagination-trained value functions
declare -A WM=([sigreg]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [linf]=lewm_dubins_safeadv_linf50
               [lam_bdet]=lewm_dubins_lam_bdet50 [jacpull_linf]=lewm_dubins_jacpull_safeadv_linf50 [jacpull]=lewm_dubins_jacpull50)
i=0; pids=()
for t in "${!WM[@]}"; do ck=${WM[$t]}
  [ -f $MDIR/$ck/margin_hj_red_rk8_s0.pt ] && continue
  CUDA_VISIBLE_DEVICES=$((i % 2)) "$LCBF/.venv/bin/python" -u "$LCBF/src/latent_cbf/scripts/train_margin_hj_lewm.py" \
    --lewm_ckpt "$ck/weights_epoch_50.pt" --out_path "$MDIR/$ck/margin_hj_red_rk8_s0.pt" --obstacle_color red \
    --n_states 12000 --rollout_k 8 --steps 4000 --seed 0 > "$LOG/margin_hj_red_rk8_${ck}.log" 2>&1 &
  pids+=($!); i=$((i + 1))
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
echo "$(date +%H:%M) rk8 heads: $(ls $MDIR/*/margin_hj_red_rk8_s0.pt 2>/dev/null | wc -l)"
# C: diagnostic on all models (rk8 heads now present for the 6 grid models)
CUDA_VISIBLE_DEVICES=1 $PY diag_rollout_safety.py baseline=sigreg_only_dubins jacobian=jacobian_w1_dubins jacpull=lewm_dubins_jacpull50 \
  sa_linf=lewm_dubins_safeadv_linf50 sa_jacpull_linf=lewm_dubins_jacpull_safeadv_linf50 lam_bdet=lewm_dubins_lam_bdet50 \
  sa_det=lewm_dubins_sa_det50 sa_enc=lewm_dubins_sa_enc50 sa_pred=lewm_dubins_sa_pred50 \
  sa_linf_s1=lewm_dubins_safeadv_linf50_s1 sa_linf_s2=lewm_dubins_safeadv_linf50_s2 sa_eps4=lewm_dubins_safeadv_eps4 \
  sa_eps16=lewm_dubins_safeadv_eps16 sa_pgd5=lewm_dubins_safeadv_pgd5 sa_rand=lewm_dubins_safeadv_rand50 \
  sa_nohead=lewm_dubins_linf_latent50 lz_det=lewm_dubins_lz_det50 lz_enc=lewm_dubins_lz_enc50 lz_pred=lewm_dubins_lz_pred50 \
  lz_rand=lewm_dubins_lz_rand50 lam_batch=lewm_dubins_lam_batch50 lam_global=lewm_dubins_lam_global50 \
  lam_gproj=lewm_dubins_lam_gproj50 sa_hue=lewm_dubins_safeadv2_50 sa_jacpull_hue=lewm_dubins_jacpull_safeadv2_50 2>&1 | grep -E "^DIAG|Error|Traceback"
# B2: safety-first CEM with the imagination-trained value functions (red)
for t in linf lam_bdet jacobian sigreg; do
  CONST=10000 ./soft_cem.sh 0 ${WM[$t]} $MDIR/${WM[$t]}/margin_hj_red_rk8_s0.pt ${t}_rk8 1000 red
done
wait; echo ABLATE_DONE
