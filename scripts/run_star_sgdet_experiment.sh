#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LOBB_RPCM_sgdet_train.sh-compatible route. The config activates the frozen
# d1/d2 pretrained OBB detector; PPG/PPN/RSGP is mandatory at evaluation.
CONFIG="${CONFIG:-configs/star_sgdet_obb_train.py}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgdet_obb_train}" \
FILTER_METHOD="${FILTER_METHOD:-PPG}" \
INIT_RPCM="${INIT_RPCM:-}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"
