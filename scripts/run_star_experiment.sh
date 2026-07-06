#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT="${EXPERIMENT:-legacy_6850}"
START_EPOCH="${START_EPOCH:-}"
RESUME="${RESUME:-}"

cd "${ROOT_DIR}"

case "${EXPERIMENT}" in
  main|legacy|legacy_6850)
    CONFIG="${CONFIG:-configs/star_predcls_obb_rpcm_legacy_ppg_6850_train.py}"
    OUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_rpcm_legacy_ppg_6850}"
    INIT_RPCM_DEFAULT="/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth"
    ;;
  legacy_17000)
    CONFIG="${CONFIG:-configs/star_predcls_obb_rpcm_legacy_ppg_train.py}"
    OUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_rpcm_legacy_ppg_17000}"
    INIT_RPCM_DEFAULT="/home/ubuntu/research/ssd/RPCM/Checkpoints/LOBB_RPCM_predcls_train/17000.pth"
    export PREDICT_USE_BIAS="${PREDICT_USE_BIAS:-1}"
    ;;
  *)
    echo "Unknown EXPERIMENT='${EXPERIMENT}'" >&2
    echo "Available: main, legacy, legacy_6850, legacy_17000" >&2
    exit 2
    ;;
esac

INIT_RPCM="${INIT_RPCM-${INIT_RPCM_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/train.log}"

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
  echo "Set INIT_RPCM='' to train from random relation-head initialization." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
STATUS_FILE="${OUT_DIR}/exit_code.txt"
PID_FILE="${OUT_DIR}/train.pid"
rm -f "${STATUS_FILE}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(python train.py --config "${CONFIG}" --device "${DEVICE}")
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
elif [[ -n "${INIT_RPCM}" ]]; then
  cmd+=(--init-rpcm "${INIT_RPCM}")
fi
if [[ -n "${START_EPOCH}" ]]; then
  cmd+=(--start-epoch "${START_EPOCH}")
fi

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

pid="$!"
printf "%s\n" "${pid}" > "${PID_FILE}"

echo "Started STAR RPCM LEGACY experiment: ${EXPERIMENT}"
echo "PID: ${pid}"
echo "Config: ${CONFIG}"
echo "Init RPCM: ${INIT_RPCM:-<none>}"
echo "Resume: ${RESUME:-<none>}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Status: ${STATUS_FILE}"
