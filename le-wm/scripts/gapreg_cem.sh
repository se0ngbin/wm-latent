#!/usr/bin/env bash
# CEM on the 15-epoch gap-keeping screen models (train_gapreg_screen.sh) and the jac15 control: decision-run protocol
# (CCEM minh, K=15 crossing goals, 200 episodes, red / purple / rot 90) with the LINEAR margin function trained on
# uniform + predicted latents (train_gp_margin_imag_dubins.py HEAD=lin -> margin_lin_imag_red_s0.pt). One lane per
# model, each starting when the model's weights_epoch_15.pt exists. Logs: run_logs/softcem_<name>_linimag_gr_..._k15_*.
# Usage: gapreg_cem.sh <gpu> name ...
set -uo pipefail
cd "$(dirname "$0")"
G=$1; shift
PY=/home/seongbin/latent/le-wm/.venv/bin/python
CK=/data/seongbin/lewm/checkpoints; M=/data/seongbin/dreamer/lewm
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4
export NEVAL=200 GOALK=15 GOALCOST=final CROSS=1 CCEM=1 CCEMRULE=minh
lane () {
  local n=$1 a=${1}_e50
  until [ -f $CK/$n/weights_epoch_15.pt ]; do sleep 120; done
  sleep 30
  if [ ! -d $CK/$a ]; then   # same alias as gapreg_screen_eval.sh
    mkdir -p $CK/$a && ln -sf $CK/$n/config.json $CK/$a/config.json && ln -sf $CK/$n/config.yaml $CK/$a/config.yaml \
      && ln -sf $CK/$n/weights_epoch_15.pt $CK/$a/weights_epoch_50.pt
  fi
  [ -f $M/$a/margin_lin_imag_red_s0.pt ] || HEAD=lin CUDA_VISIBLE_DEVICES=$G $PY train_gp_margin_imag_dubins.py $a 0 2>&1 | grep -E "saved|Error"
  ./soft_cem.sh $G $a $M/$a/margin_lin_imag_red_s0.pt ${n#lewm_dubins_}_linimag_gr 1000 red purple fullrot90
}
for n in "$@"; do lane $n & sleep 5; done
wait; echo GAPREG_CEM_DONE
