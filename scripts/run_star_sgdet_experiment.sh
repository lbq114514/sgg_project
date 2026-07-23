#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LOBB_RPCM_sgdet_train.sh-compatible route. The config activates the frozen
# d1/d2 pretrained OBB detector; PPG/PPN/RSGP is mandatory at evaluation.
# The detector is frozen and its full-resolution patch pass is prohibitively
# slow inside the training loop, so the sgdet launcher uses the prebuilt,
# read-only detection cache by default.  Set
# SGDET_DETECTION_CACHE_ENABLED=0 explicitly only for detector-path debugging.
#
# Common override examples:
#   RESUME=outputs/.../model_last.pth bash scripts/run_star_sgdet_experiment.sh
#   FILTER_METHOD=PPG bash scripts/run_star_sgdet_experiment.sh
#   PRINT_TRAIN_STEP_FREQ=10 bash scripts/run_star_sgdet_experiment.sh
#
# Before formal training, both train and test manifests must exist in the
# cache directory. The launcher defaults to REQUIRE_HIT=1, so a missing file
# or hash mismatch fails immediately instead of silently running the slow
# detector.
CONFIG="${CONFIG:-configs/star_sgdet_obb_train.py}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgdet_obb_train}" \
FILTER_METHOD="${FILTER_METHOD:-RSGP}" \
RSGP_MODE="${RSGP_MODE:-HYBRID}" \
RSGP_TOPK="${RSGP_TOPK:-10000}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}" \
SGDET_TRAIN_LABEL_SOURCE="${SGDET_TRAIN_LABEL_SOURCE:-matched_gt}" \
SGDET_FILTER_LABEL_SOURCE="${SGDET_FILTER_LABEL_SOURCE:-matched_gt}" \
ADD_GTBOX_TO_PROPOSAL_IN_TRAIN="${ADD_GTBOX_TO_PROPOSAL_IN_TRAIN:-0}" \
SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}" \
SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-outputs/star_sgdet_detection_cache}" \
SGDET_DETECTION_CACHE_REQUIRE_HIT="${SGDET_DETECTION_CACHE_REQUIRE_HIT:-1}" \
SGDET_DETECTION_CACHE_HASH="${SGDET_DETECTION_CACHE_HASH:-}" \
INIT_RPCM="${INIT_RPCM:-}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"
