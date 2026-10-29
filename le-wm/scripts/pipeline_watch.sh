#!/usr/bin/env bash
# Start pipeline_new_wm.sh for each pending Dubins WM as soon as its weights_epoch_50.pt exists (one pipeline per model,
# alternating GPUs). sa_enc50 (adv_h encoder-only) already exists and starts first.
cd "$(dirname "$0")"
declare -A NAME=(
  [lewm_dubins_sa_enc50]="sa_enc|adv_h enc-only"
  [lewm_dubins_jac_safeadv_linf50]="jac_linf|jac+adv_h"
  [lewm_dubins_jacpull_lam_bdet50]="jacpull_lam_bdet|jac+pull+adv_λ"
  [lewm_dubins_jac_safeadv_linf_enc50]="jac_linf_enc|jac+adv_h enc-only"
  [lewm_dubins_jacpull_safeadv_linf_enc50]="jacpull_linf_enc|jac+pull+adv_h enc-only"
  [lewm_dubins_lam_bdet_enc50]="lam_bdet_enc|adv_λ enc-only"
  [lewm_dubins_jac_lam_bdet_enc50]="jac_lam_bdet_enc|jac+adv_λ enc-only"
  [lewm_dubins_jacpull_lam_bdet_enc50]="jacpull_lam_bdet_enc|jac+pull+adv_λ enc-only"
)
declare -A STARTED=(); g=0
while [ ${#STARTED[@]} -lt ${#NAME[@]} ]; do
  for ck in "${!NAME[@]}"; do
    [ -n "${STARTED[$ck]:-}" ] && continue
    [ -f "/data/seongbin/lewm/checkpoints/$ck/weights_epoch_50.pt" ] || continue
    sleep 120   # let the final checkpoint finish writing
    IFS='|' read -r tag name <<< "${NAME[$ck]}"
    echo "$(date +%H:%M) start pipeline $tag (gpu $g)"
    ./pipeline_new_wm.sh "$ck" "$tag" "$name" $g > "/home/seongbin/latent/run_logs/pipeline_$tag.log" 2>&1 &
    STARTED[$ck]=1; g=$((1 - g))
  done
  sleep 300
done
wait; echo PIPELINE_WATCH_DONE
