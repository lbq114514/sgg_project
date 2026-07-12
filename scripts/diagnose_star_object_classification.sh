#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/star_sgcls_obb_train.py}"
CHECKPOINT="${CHECKPOINT:-outputs/star_sgcls_obb_train/model_last.pth}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
TOPK="${TOPK:-2000}"
OUTPUT="${OUTPUT:-}"
FILTER_METHOD="${FILTER_METHOD:-RSGP}"
RSGP_MODE="${RSGP_MODE:-HYBRID}"
RSGP_TOPK="${RSGP_TOPK:-10000}"
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(
  python tools/diagnose_object_classification.py
  --config "${CONFIG}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --max-images "${MAX_IMAGES}"
  --topk "${TOPK}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${OUTPUT}" ]]; then
  cmd+=(--output "${OUTPUT}")
fi

echo "Object classification diagnostics"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT:-<none>}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "Max images: ${MAX_IMAGES}"
echo "Filter: ${FILTER_METHOD}"
"${cmd[@]}"
