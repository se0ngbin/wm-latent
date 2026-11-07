#!/usr/bin/env bash
# Goal-distance sweep (goal_dist_sweep.sh, CCEM minh rule) on the extra world-model seeds that are already trained:
# K 10/15 for all first, then 20/25; raw, CCEM-MLP, CCEM-linear; red / purple / rot 90; 50 episodes. 4 lanes, all on GPU0 (GPU1 is full; GPUs are
# shared with other jobs). jac uses seeds 1 and 100 (paper grid: 3072, 1, 100).
cd "$(dirname "$0")"
export CCEMRULE=minh
D=lewm_dubins_
exec ./goal_dist_sweep.sh \
 "0|sigreg_only_dubins_s1:sigreg_s1,sigreg_only_dubins_s2:sigreg_s2,${D}safeadv_linf50_s1:linf_s1,${D}safeadv_linf50_s2:linf_s2" \
 "0|${D}lam_bdet50_s1:lam_bdet_s1,${D}lam_bdet50_s2:lam_bdet_s2,jacobian_w1_dubins_s1:jacobian_s1,${D}jacw1_s100:jacobian_s100" \
 "0|${D}jac_safeadv_linf_enc50_s1:jac_linf_enc_s1,${D}jac_safeadv_linf_enc50_s2:jac_linf_enc_s2,${D}jacpull50_s1:jacpull_s1,${D}jacpull50_s2:jacpull_s2" \
 "0|${D}jacpull_safeadv_linf50_s1:jacpull_linf_s1,${D}jacpull_safeadv_linf50_s2:jacpull_linf_s2,${D}jacpull_safeadv_linf_enc50_s1:jacpull_linf_enc_s1,${D}jac_lam_bdet50_s1:jac_lam_bdet_s1,${D}jacpull_lam_bdet50_s1:jacpull_lam_bdet_s1,${D}jac_lam_bdet_enc50_s1:jac_lam_bdet_enc_s1"
