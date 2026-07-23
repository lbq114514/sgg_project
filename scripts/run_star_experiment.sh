#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sgg}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/star_predcls_obb_train.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_train}"
INIT_RPCM="${INIT_RPCM:-}"
START_EPOCH="${START_EPOCH:-}"
RESUME="${RESUME:-}"
RESET_SOLVER_ON_RESUME="${RESET_SOLVER_ON_RESUME:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

cd "${ROOT_DIR}"

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

mkdir -p "${OUTPUT_DIR}"
STATUS_FILE="${OUTPUT_DIR}/exit_code.txt"
PID_FILE="${OUTPUT_DIR}/train.pid"
rm -f "${STATUS_FILE}" "${PID_FILE}"

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
if [[ -n "${RESET_SOLVER_ON_RESUME}" && "${RESET_SOLVER_ON_RESUME}" != "0" ]]; then
  cmd+=(--reset-solver-on-resume)
fi

if [[ -n "${RESUME}" ]]; then
  nohup bash -lc '
    set +e
    "$@" &
    child_pid=$!
    printf "%s\n" "${child_pid}" > "'"${PID_FILE}"'"
    wait "${child_pid}"
    status=$?
    printf "%s\n" "${status}" > "'"${STATUS_FILE}"'"
    exit "${status}"
  ' _ "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
else
  nohup bash -lc '
    set +e
    "$@" &
    child_pid=$!
    printf "%s\n" "${child_pid}" > "'"${PID_FILE}"'"
    wait "${child_pid}"
    status=$?
    printf "%s\n" "${status}" > "'"${STATUS_FILE}"'"
    exit "${status}"
  ' _ "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
fi

wrapper_pid="$!"
pid="${wrapper_pid}"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -s "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    break
  fi
  sleep 0.1
done

echo "Started STAR main experiment"
echo "PID: ${pid}"
echo "Wrapper PID: ${wrapper_pid}"
echo "Config: ${CONFIG}"
echo "Init RPCM: ${INIT_RPCM:-<none>}"
echo "Resume: ${RESUME:-<none>}"
echo "Reset solver on resume: ${RESET_SOLVER_ON_RESUME:-0}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Status: ${STATUS_FILE}"
