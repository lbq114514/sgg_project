#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_train.py}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-outputs/star_predcls_obb_train/model_last.pth}"
LOG_FILE="${LOG_FILE:-star_train_resume.log}"
START_EPOCH=44

cd "${ROOT_DIR}"
if [[ ! -f "${RESUME}" ]]; then
  echo "Checkpoint not found: ${RESUME}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

nohup python train.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --resume "${RESUME}" \
  --start-epoch "${START_EPOCH}" \
  >> "${LOG_FILE}" 2>&1 &

echo "Resumed training with PID $!"
echo "Checkpoint: ${RESUME}"
echo "Log: ${LOG_FILE}"
