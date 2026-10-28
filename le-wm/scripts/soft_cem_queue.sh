#!/usr/bin/env bash
# Everything in one queue: 4 serial lanes (2 per GPU). Weight sweep + lexicographic on sigreg/jacobian (red), the
# adv-λ plain-CEM baseline (w=0), then the lexicographic grid (6 models x 4 conditions). soft_cem.sh skips any cell
# whose log already has a result, so the grid's red cells for sigreg/jacobian reuse the lex runs.
cd "$(dirname "$0")"
H=/data/seongbin/dreamer/lewm
hd () { echo "$H/$1/margin_hj_color_red_rk0_s0.pt"; }
ALL="red purple diamond fullrot90"
lex () { CONST=10000 ./soft_cem.sh "$1" "$2" "$(hd "$2")" "$3" 1000 "${@:4}"; }
{ ./soft_cem.sh 0 jacobian_w1_dubins "$(hd jacobian_w1_dubins)" jacobian 100 red
  ./soft_cem.sh 0 jacobian_w1_dubins "$(hd jacobian_w1_dubins)" jacobian 10000 red
  lex 0 jacobian_w1_dubins jacobian $ALL
  lex 0 lewm_dubins_jacpull_safeadv_linf50 jacpull_linf $ALL; } &
{ ./soft_cem.sh 0 jacobian_w1_dubins "$(hd jacobian_w1_dubins)" jacobian 1000 red
  ./soft_cem.sh 0 lewm_dubins_lam_bdet50 "$(hd lewm_dubins_lam_bdet50)" lam_bdet 0 $ALL
  lex 0 lewm_dubins_jacpull50 jacpull $ALL; } &
{ ./soft_cem.sh 1 sigreg_only_dubins "$(hd sigreg_only_dubins)" sigreg 100 red
  ./soft_cem.sh 1 sigreg_only_dubins "$(hd sigreg_only_dubins)" sigreg 10000 red
  lex 1 sigreg_only_dubins sigreg $ALL
  lex 1 lewm_dubins_lam_bdet50 lam_bdet red purple; } &
{ ./soft_cem.sh 1 sigreg_only_dubins "$(hd sigreg_only_dubins)" sigreg 1000 red
  lex 1 lewm_dubins_safeadv_linf50 linf $ALL
  lex 1 lewm_dubins_lam_bdet50 lam_bdet diamond fullrot90; } &
wait; echo QUEUE_DONE
