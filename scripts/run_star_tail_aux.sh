#!/usr/bin/env bash
set -euo pipefail

# Clean detector-only HPRC training launcher.
#
# The implementation/config keys retain the historical ``tail_aux`` name for
# checkpoint compatibility.  A fresh launch initializes only the frozen OBB
# detector configured by star_predcls_obb_tail_aux_train.py; relation/GNN,
# prototype and HPRC parameters start from their corrected initializers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-configs/star_predcls_obb_tail_aux_train.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_hprc_scratch}"
DETECTOR_CHECKPOINT="pretrained/OBB_swin_L_OBD.pth"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DETECTOR_CHECKPOINT}" ]]; then
  echo "Detector checkpoint not found: ${DETECTOR_CHECKPOINT}" >&2
  exit 1
fi

echo "Launching clean STAR PredCls HPRC training"
echo "Config: ${CONFIG}"
echo "Initialization: detector-only (${DETECTOR_CHECKPOINT})"
echo "Relation initialization: corrected RPCM/HPRC initializers (no RPCM checkpoint)"
echo "Output: ${OUTPUT_DIR}"

# Force INIT_RPCM empty even if it exists in the caller's environment.  Resume
# remains available through RESUME=... and is handled by the common launcher.
CONFIG="${CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
INIT_RPCM="" \
BASE_LR="${HPRC_BASE_LR:-0.016}" \
MAX_EPOCHS="${HPRC_MAX_EPOCHS:-300}" \
STEPS="${HPRC_STEPS:-10000,14000,16000}" \
VAL_START_PERIOD="${HPRC_VAL_START_PERIOD:-120}" \
VAL_PERIOD="${HPRC_VAL_PERIOD:-2}" \
CHECKPOINT_PERIOD="${HPRC_CHECKPOINT_PERIOD:-4}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"
