#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_train.py}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_eval_legacy}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/test.log}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/test_metrics.json}"
FILTER_METHOD="${FILTER_METHOD:-PPG}"
PAIR_FILTER_CHECKPOINT="${PAIR_FILTER_CHECKPOINT:-}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-legacy-rpcm}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(
  python tools/eval_once.py
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --checkpoint-load-mode "${CHECKPOINT_LOAD_MODE}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --output "${OUTPUT_JSON}"
)

if [[ -n "${FILTER_METHOD}" ]]; then
  cmd+=(--filter-method "${FILTER_METHOD}")
fi
if [[ -n "${PAIR_FILTER_CHECKPOINT}" ]]; then
  cmd+=(--pair-filter-checkpoint "${PAIR_FILTER_CHECKPOINT}")
fi

nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started evaluation with PID $!"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Split: ${SPLIT}"
echo "Filter: ${FILTER_METHOD}"
echo "Load mode: ${CHECKPOINT_LOAD_MODE}"
echo "Log: ${LOG_FILE}"
echo "JSON: ${OUTPUT_JSON}"
