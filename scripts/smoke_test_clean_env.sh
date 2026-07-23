#!/usr/bin/env bash
set -euo pipefail

# Small end-to-end verification for a newly installed environment. This runs
# in the currently active Python environment and evaluates only a deterministic
# prefix of the STAR test split.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/star_predcls_obb_tail_aux_train.py}"
CHECKPOINT="${CHECKPOINT:-outputs/star_predcls_obb_tail_aux/best.pth}"
FILTER_METHOD="${FILTER_METHOD:-PPG}"
MAX_IMAGES="${MAX_IMAGES:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/environment_smoke_test}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/test_metrics.json}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Smoke-test config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Smoke-test checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[1/2] Validating Python, CUDA, and mmcv-full rotated operators"
"${PYTHON_BIN}" tools/check_environment.py --strict --require-cuda

echo "[2/2] Running STAR PredCls on ${MAX_IMAGES} test images"
export TEST_BATCH_SIZE=1
export VAL_BATCH_SIZE=1
export FILTER_METHOD
"${PYTHON_BIN}" tools/eval_once.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-load-mode full \
  --split test \
  --device cuda \
  --filter-method "${FILTER_METHOD}" \
  --max-images "${MAX_IMAGES}" \
  --output "${OUTPUT_JSON}"

if [[ ! -s "${OUTPUT_JSON}" ]]; then
  echo "Smoke test did not produce metrics: ${OUTPUT_JSON}" >&2
  exit 1
fi

echo
echo "Smoke test passed. Metrics: ${OUTPUT_JSON}"

