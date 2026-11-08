#!/usr/bin/env bash
# Goal-distance sweep (goal_dist_sweep.sh, CCEM minh rule) on the extra world-model seeds of the paper set
# (1a, 2a, 2b, 2d, 2f, 3a, 3b, 3d, 3f; dropped families are not run): goal distances 10, 15, 20 in one phase (25 is
# archived); raw, CCEM-MLP, CCEM-linear; red / purple / rot 90; 50 episodes. 5 lanes. Finished cells are skipped.
# jac uses seeds 1 and 100 (paper grid: 3072, 1, 100). 2b has no extra seeds trained yet.
cd "$(dirname "$0")"
export CCEMRULE=minh PHASES="10 15 20"
D=lewm_dubins_
exec ./goal_dist_sweep.sh \
 "0|sigreg_only_dubins_s1:sigreg_s1,sigreg_only_dubins_s2:sigreg_s2,${D}jac_lam_bdet50_s1:jac_lam_bdet_s1" \
 "1|jacobian_w1_dubins_s1:jacobian_s1,${D}jacw1_s100:jacobian_s100,${D}jacpull_lam_bdet50_s1:jacpull_lam_bdet_s1" \
 "1|${D}jac_safeadv_linf_enc50_s1:jac_linf_enc_s1,${D}jac_safeadv_linf_enc50_s2:jac_linf_enc_s2,${D}jacpull_safeadv_linf_enc50_s1:jacpull_linf_enc_s1" \
 "1|${D}jacpull50_s1:jacpull_s1,${D}jacpull50_s2:jacpull_s2" \
 "1|${D}jacpull_safeadv_linf50_s1:jacpull_linf_s1,${D}jacpull_safeadv_linf50_s2:jacpull_linf_s2"
