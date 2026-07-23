#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sgg}"
CONFIG="${CONFIG:-configs/star_sgdet_obb_train.py}"
SPLITS="${SPLITS:-train,test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgdet_detection_cache}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_IMAGES="${MAX_IMAGES:--1}"
OVERWRITE="${OVERWRITE:-1}"

# This script runs only the frozen detector and writes relation-view proposals.
# It neither trains the relation head nor caches GT relations. MAX_IMAGES can
# be used for a 2/32-image smoke build first:
#   MAX_IMAGES=32 SPLITS=test bash scripts/build_sgdet_detection_cache.sh

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(
  python tools/build_sgdet_detection_cache.py
  --config "${CONFIG}"
  --split "${SPLITS}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-images "${MAX_IMAGES}"
)
if [[ "${OVERWRITE}" != "0" ]]; then
  cmd+=(--overwrite)
fi

echo "Building sgdet detection cache"
echo "Config: ${CONFIG}"
echo "Splits: ${SPLITS}"
echo "Output: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Max images: ${MAX_IMAGES}"
echo "Overwrite: ${OVERWRITE}"

"${cmd[@]}"
