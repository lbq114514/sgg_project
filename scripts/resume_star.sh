#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_rpcm_legacy_ppg_6850_train.py}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_rpcm_legacy_ppg_6850}"
RESUME="${RESUME:-${OUTPUT_DIR}/model_last.pth}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/resume.log}"
START_EPOCH="${START_EPOCH:-}"

cd "${ROOT_DIR}"
if [[ ! -f "${RESUME}" ]]; then
  echo "Checkpoint not found: ${RESUME}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(python train.py --config "${CONFIG}" --device "${DEVICE}" --resume "${RESUME}")
if [[ -n "${START_EPOCH}" ]]; then
  cmd+=(--start-epoch "${START_EPOCH}")
fi

nohup "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &

echo "Resumed training with PID $!"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${RESUME}"
echo "Log: ${LOG_FILE}"
