#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LOBB_RPCM_sgcls_train.sh-compatible route.  OBB_swin_L_OBD.pth is loaded
# inside the config; do not set INIT_RPCM unless explicitly studying transfer.
CONFIG="${CONFIG:-configs/star_sgcls_obb_train.py}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgcls_obb_train}" \
FILTER_METHOD="${FILTER_METHOD:-PPG}" \
INIT_RPCM="${INIT_RPCM:-}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"
