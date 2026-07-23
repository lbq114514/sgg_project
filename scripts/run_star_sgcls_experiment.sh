#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LOBB_RPCM_sgcls_train.sh-compatible route.  OBB_swin_L_OBD.pth is loaded
# inside the config; do not set INIT_RPCM unless explicitly studying transfer.
# SGCLS_FILTER_LABEL_SOURCE=gt matches the SGG-Toolkit STAR pair-filter
# protocol. Prefix the command with SGCLS_FILTER_LABEL_SOURCE=pred for a strict
# predicted-label ablation.
CONFIG="${CONFIG:-configs/star_sgcls_obb_train.py}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgcls_obb_train}" \
FILTER_METHOD="${FILTER_METHOD:-RSGP}" \
RSGP_MODE="${RSGP_MODE:-HYBRID}" \
RSGP_TOPK="${RSGP_TOPK:-10000}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}" \
INIT_RPCM="${INIT_RPCM:-}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"
