#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
FILTER_METHOD="${FILTER_METHOD:-}"
PAIR_FILTER_CHECKPOINT="${PAIR_FILTER_CHECKPOINT:-}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-full}"

usage() {
  cat >&2 <<'EOF'
Usage:
  CONFIG=... CHECKPOINT=... OUTPUT_DIR=... [options] bash scripts/eval_once.sh

Required:
  CONFIG       Python config used to build the model.
  CHECKPOINT   Checkpoint to evaluate.
  OUTPUT_DIR   Directory for test.log and test_metrics.json.

Common options:
  SPLIT=test|val                         default: test
  DEVICE=cuda|cpu                        default: cuda
  CHECKPOINT_LOAD_MODE=full|model-only|legacy-rpcm
                                        default: full
  FILTER_METHOD=PPG|PPN|RSGP             optional runtime override;
                                        if omitted, the config decides.
  PAIR_FILTER_CHECKPOINT=...             optional PPG/PPN checkpoint override.
  LOG_FILE=...                           default: ${OUTPUT_DIR}/test.log
  OUTPUT_JSON=...                        default: ${OUTPUT_DIR}/test_metrics.json

This script intentionally has no default CONFIG/CHECKPOINT/OUTPUT_DIR to avoid
silently evaluating a checkpoint with the wrong model config.
EOF
}

cd "${ROOT_DIR}"

if [[ -z "${CONFIG:-}" || -z "${CHECKPOINT:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  usage
  exit 2
fi

LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/test.log}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/test_metrics.json}"

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
echo "Filter override: ${FILTER_METHOD:-<config>}"
echo "Pair filter checkpoint override: ${PAIR_FILTER_CHECKPOINT:-<config>}"
echo "Load mode: ${CHECKPOINT_LOAD_MODE}"
echo "Log: ${LOG_FILE}"
echo "JSON: ${OUTPUT_JSON}"
