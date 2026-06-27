#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT="${EXPERIMENT:-main}"
START_EPOCH="${START_EPOCH:-}"
RESUME="${RESUME:-}"

cd "${ROOT_DIR}"

case "${EXPERIMENT}" in
  main)
    CONFIG="configs/star_predcls_obb_train.py"
    OUT_DIR="outputs/star_predcls_obb_train"
    LOG_FILE="star_train_main.log"
    ;;
  pairness_score)
    CONFIG="configs/star_predcls_obb_ablate_no_ppg.py"
    OUT_DIR="outputs/star_predcls_obb_ablate_no_ppg"
    LOG_FILE="star_train_no_ppg.log"
    ;;
  no_proto_bias)
    CONFIG="configs/star_predcls_obb_ablate_no_proto.py"
    OUT_DIR="outputs/star_predcls_obb_ablate_no_proto"
    LOG_FILE="star_train_no_proto.log"
    ;;
  pairness_tight)
    CONFIG="configs/star_predcls_obb_ablate_pairness_tight.py"
    OUT_DIR="outputs/star_predcls_obb_ablate_pairness_tight"
    LOG_FILE="star_train_pairness_tight.log"
    ;;
  random_proto)
    CONFIG="configs/star_predcls_obb_ablate_random_proto.py"
    OUT_DIR="outputs/star_predcls_obb_ablate_random_proto"
    LOG_FILE="star_train_random_proto.log"
    ;;
  tail_sampling)
    CONFIG="configs/star_predcls_obb_ablate_tail_sampling.py"
    OUT_DIR="outputs/star_predcls_obb_ablate_tail_sampling"
    LOG_FILE="star_train_tail_sampling.log"
    ;;
  *)
    echo "Unknown EXPERIMENT='${EXPERIMENT}'" >&2
    echo "Available: main, pairness_score, no_proto_bias, pairness_tight, random_proto, tail_sampling" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"
STATUS_FILE="${OUT_DIR}/exit_code.txt"
rm -f "${STATUS_FILE}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cmd=(python train.py --config "${CONFIG}" --device "${DEVICE}")
if [[ -n "${RESUME}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "Checkpoint not found: ${RESUME}" >&2
    exit 1
  fi
  cmd+=(--resume "${RESUME}")
fi
if [[ -n "${START_EPOCH}" ]]; then
  cmd+=(--start-epoch "${START_EPOCH}")
fi

# nohup bash -lc '
#   "$@"
#   status=$?
#   printf "%s\n" "${status}" > "'"${STATUS_FILE}"'"
#   exit "${status}"
# ' _ "${cmd[@]}" > "${LOG_FILE}" 2>&1 &

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
echo "${pid}" > "${OUT_DIR}/train.pid"

echo "Started STAR experiment: ${EXPERIMENT}"
echo "PID: ${pid}"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Status: ${STATUS_FILE}"
