#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pyg}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/home/ubuntu/research/ssd/RPCM}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml}"
REFERENCE_MM_CONFIG="${REFERENCE_MM_CONFIG:-configs/RSOBB/STAR_obb_predcls_sgcls.py}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth}"
CURRENT_CONFIG="${CURRENT_CONFIG:-configs/star_sgcls_obb_train.py}"
IMAGE_ID="${IMAGE_ID:-4}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sgcls_parity_probe}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${REFERENCE_ROOT}" ]]; then
  echo "Reference runtime not found: ${REFERENCE_ROOT}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${ROOT_DIR}"

python scripts/dump_rpcm_original_probe.py \
  --reference-root "${REFERENCE_ROOT}" \
  --config-file "${REFERENCE_CONFIG}" \
  --mm-config "${REFERENCE_MM_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --task sgcls \
  --detector-only \
  --image-id "${IMAGE_ID}" \
  --device "${DEVICE}" \
  --output "${ROOT_DIR}/${OUTPUT_DIR}/reference_sgcls.pt"

for source in gt pred; do
  python scripts/dump_rpcm_current_probe.py \
    --config "${CURRENT_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --sgcls-filter-label-source "${source}" \
    --detector-only \
    --image-id "${IMAGE_ID}" \
    --reference-dump "${ROOT_DIR}/${OUTPUT_DIR}/reference_sgcls.pt" \
    --device "${DEVICE}" \
    --output "${OUTPUT_DIR}/current_sgcls_${source}.pt"

  python scripts/compare_rpcm_probe_dumps.py \
    --left "${OUTPUT_DIR}/reference_sgcls.pt" \
    --right "${OUTPUT_DIR}/current_sgcls_${source}.pt" \
    > "${OUTPUT_DIR}/compare_reference_vs_current_${source}.log"
done

echo "sgcls parity probe complete: ${OUTPUT_DIR}"
