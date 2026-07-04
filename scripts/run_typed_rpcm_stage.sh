#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE="${STAGE:-stage1}"
CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-}"
START_EPOCH="${START_EPOCH:-}"
INIT_RPCM="${INIT_RPCM-outputs/star_predcls_obb_train_large_long/model_last.pth}"

[[ "${STAGE}" == "stage1" ]] || { echo "Only STAGE=stage1 is supported" >&2; exit 2; }
CONFIG="configs/star_predcls_obb_typed_stage1_train.py"
OUT_DIR="outputs/star_predcls_obb_typed_stage1"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/train.log}"
PID_FILE="${OUT_DIR}/train.pid"
STATUS_FILE="${OUT_DIR}/exit_code.txt"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "Resume checkpoint not found: ${RESUME}" >&2
  exit 1
fi
if [[ -z "${RESUME}" && -n "${INIT_RPCM}" && ! -f "${INIT_RPCM}" ]]; then
  echo "RPCM initialization checkpoint not found: ${INIT_RPCM}" >&2
  echo "Set INIT_RPCM='' to train without RPCM initialization." >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(python train.py --config "${CONFIG}" --device "${DEVICE}")
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
else
  [[ -n "${INIT_RPCM}" ]] && cmd+=(--init-rpcm "${INIT_RPCM}")
fi
[[ -n "${START_EPOCH}" ]] && cmd+=(--start-epoch "${START_EPOCH}")

rm -f "${STATUS_FILE}"
if [[ -n "${RESUME}" ]]; then
  nohup bash -lc '
    "$@"
    status=$?
    printf "%s\n" "${status}" > "'"${STATUS_FILE}"'"
    exit "${status}"
  ' _ "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
else
  nohup bash -lc '
    "$@"
    status=$?
    printf "%s\n" "${status}" > "'"${STATUS_FILE}"'"
    exit "${status}"
  ' _ "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
fi

pid=$!
printf "%s\n" "${pid}" > "${PID_FILE}"
echo "Started Typed RPCM Stage1"
echo "PID: ${pid}"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Status: ${STATUS_FILE}"
