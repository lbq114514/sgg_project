#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_train.py}"
CHECKPOINT="${CHECKPOINT:-outputs/star_predcls_obb_train_large_long/model_last.pth}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
LOG_FILE="${LOG_FILE:-star_eval_once.log}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(
  python tools/eval_once.py
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --device "${DEVICE}"
)

nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started evaluation with PID $!"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Split: ${SPLIT}"
echo "Log: ${LOG_FILE}"
