#!/usr/bin/env bash
# The goal-distance sweep for every finished seed-3072 Dubins world model (17 of 21; the 4 untrained linear-head rows are
# missing): 9 lanes, K 10/15 for all first, then 20/25 (goal_dist_sweep.sh).
cd "$(dirname "$0")"
D=lewm_dubins_
exec ./goal_dist_sweep.sh \
 "1|sigreg_only_dubins:sigreg,${D}safeadv_linf50:linf" \
 "1|${D}lam_bdet50:lam_bdet,${D}lam_bdet_enc50:lam_bdet_enc" \
 "1|${D}sa_enc50:sa_enc,jacobian_w1_dubins:jacobian" \
 "1|${D}jac_safeadv_linf50:jac_linf,${D}jac_safeadv_linf_linhead50_e50:jac_linf_linhead" \
 "1|${D}jac_lam_bdet50:jac_lam_bdet,${D}jac_lam_bdet_enc50:jac_lam_bdet_enc" \
 "0|${D}jac_safeadv_linf_enc50:jac_linf_enc,${D}jacpull50:jacpull" \
 "0|${D}jacpull_safeadv_linf50:jacpull_linf,${D}jacpull_lam_bdet50:jacpull_lam_bdet" \
 "0|${D}jacpull_lam_bdet_enc50:jacpull_lam_bdet_enc,${D}jacpull_safeadv_linf_enc50:jacpull_linf_enc" \
 "0|${D}jacpull_safeadv_linf_enc_linhead50_e50:jacpull_linf_enc_linhead"
