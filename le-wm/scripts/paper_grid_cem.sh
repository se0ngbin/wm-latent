#!/usr/bin/env bash
# Launch the paper-grid CEM (paper-results.tex): 21 Dubins models x 3 WM seeds, raw + safety-first, open loop, red /
# purple / rotated 90°, 50 episodes per seed. Three lanes (rows 1x, 2x, 3x); within a lane, finished WMs first, WMs still
# training or queued last (seed_cem_queue.sh waits for their epoch-50 checkpoint). jac uses seeds 3072, 1, 100.
set -uo pipefail
cd "$(dirname "$0")"; L=/home/seongbin/latent/run_logs
D=lewm_dubins
GPU=1 ./seed_cem_queue.sh \
  sigreg_only_dubins:sigreg sigreg_only_dubins_s1:sigreg_s1 sigreg_only_dubins_s2:sigreg_s2 \
  ${D}_safeadv_linf50:linf ${D}_safeadv_linf50_s1:linf_s1 ${D}_safeadv_linf50_s2:linf_s2 \
  ${D}_lam_bdet50:lam_bdet ${D}_lam_bdet50_s1:lam_bdet_s1 ${D}_lam_bdet50_s2:lam_bdet_s2 \
  ${D}_lam_bdet_enc50:lam_bdet_enc ${D}_sa_enc50:sa_enc \
  ${D}_safeadv_linf_linhead50:linf_linhead ${D}_sa_enc_linhead50:sa_enc_linhead > $L/paper_grid_cem_1.log 2>&1 &
GPU=1 ./seed_cem_queue.sh \
  jacobian_w1_dubins:jacobian jacobian_w1_dubins_s1:jacobian_s1 ${D}_jacw1_s100:jacobian_s100 \
  ${D}_jac_safeadv_linf_enc50:jac_linf_enc ${D}_jac_safeadv_linf_enc50_s1:jac_linf_enc_s1 ${D}_jac_safeadv_linf_enc50_s2:jac_linf_enc_s2 \
  ${D}_jac_safeadv_linf50:jac_linf ${D}_jac_lam_bdet50:jac_lam_bdet ${D}_jac_lam_bdet_enc50:jac_lam_bdet_enc \
  ${D}_jac_lam_bdet50_s1:jac_lam_bdet_s1 ${D}_jac_safeadv_linf_linhead50:jac_linf_linhead ${D}_jac_lam_bdet_enc50_s1:jac_lam_bdet_enc_s1 \
  ${D}_jac_safeadv_linf50_s1:jac_linf_s1 ${D}_jac_safeadv_linf50_s2:jac_linf_s2 ${D}_jac_lam_bdet50_s2:jac_lam_bdet_s2 \
  ${D}_jac_lam_bdet_enc50_s2:jac_lam_bdet_enc_s2 ${D}_jac_safeadv_linf_enc_linhead50:jac_linf_enc_linhead > $L/paper_grid_cem_2.log 2>&1 &
GPU=0 ./seed_cem_queue.sh \
  ${D}_jacpull50:jacpull ${D}_jacpull50_s1:jacpull_s1 ${D}_jacpull50_s2:jacpull_s2 \
  ${D}_jacpull_safeadv_linf50:jacpull_linf ${D}_jacpull_safeadv_linf50_s1:jacpull_linf_s1 ${D}_jacpull_safeadv_linf50_s2:jacpull_linf_s2 \
  ${D}_jacpull_lam_bdet50:jacpull_lam_bdet ${D}_jacpull_lam_bdet_enc50:jacpull_lam_bdet_enc \
  ${D}_jacpull_safeadv_linf_enc50:jacpull_linf_enc ${D}_jacpull_safeadv_linf_enc50_s1:jacpull_linf_enc_s1 \
  ${D}_jacpull_lam_bdet50_s1:jacpull_lam_bdet_s1 ${D}_jacpull_safeadv_linf_enc_linhead50:jacpull_linf_enc_linhead \
  ${D}_jacpull_safeadv_linf_linhead50:jacpull_linf_linhead ${D}_jacpull_lam_bdet_enc50_s1:jacpull_lam_bdet_enc_s1 \
  ${D}_jacpull_lam_bdet_enc50_s2:jacpull_lam_bdet_enc_s2 ${D}_jacpull_lam_bdet50_s2:jacpull_lam_bdet_s2 \
  ${D}_jacpull_safeadv_linf_enc50_s2:jacpull_linf_enc_s2 > $L/paper_grid_cem_3.log 2>&1 &
wait; echo PAPER_GRID_CEM_DONE
