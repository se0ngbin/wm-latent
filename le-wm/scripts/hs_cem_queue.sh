#!/usr/bin/env bash
# Short-horizon rank-normalized safety cost: h_steps = 5 (only the executed window; under purple h ranks imagined latents
# for ~5 steps then the predictor drifts to chance), h_quantile = 0.05 (purple collapses h's scale but not its ranking).
# Old GP h (best short-horizon purple ranking: AUC 0.80-0.88 at steps 1-5 on jac+pull+adv_h encoder-only), jac reference.
set -uo pipefail
cd "$(dirname "$0")"
export TRACE=/data/seongbin/lewm/safeadv_results/collision_trace GOALCOST=mean REC=5 CONST=10000 HQ=0.05 HSTEPS=5
H=/data/seongbin/dreamer/lewm
hs () { ./soft_cem.sh $1 $2 $H/$2/margin_gp_fail_red_s0.pt $3 1000 purple red; }
hs 0 lewm_dubins_jacpull_safeadv_linf_enc50 jacpull_linf_enc_gp_gr_trace &
hs 1 jacobian_w1_dubins jacobian_gp_gr_trace &
wait; echo HS_DONE
