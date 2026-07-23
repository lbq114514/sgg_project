#!/usr/bin/env bash
set -euo pipefail

# Inference-only pair-filter/component ablation using one fixed relation-head
# checkpoint. Run one case at a time to avoid concurrent GPU evaluations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE="${1:-}"

FILTER_METHOD=RSGP
RSGP_MODE=HYBRID
RSGP_USE_PPN_COMPLETION=1
RSGP_USE_GEOMETRY=1
RSGP_USE_ANCHOR=1
RSGP_USE_TOPOLOGY=1
RSGP_USE_TAIL_PRIOR=1
RSGP_USE_DEGREE_SCORE=1
RSGP_ENFORCE_DEGREE_CAP=1
RSGP_ENFORCE_LABEL_QUOTA=1
RSGP_RS_POOL_TOPK=12000

case "${CASE^^}" in
  PPG)
    FILTER_METHOD=PPG
    NAME="ppg"
    ;;
  PPN)
    FILTER_METHOD=PPN
    NAME="ppn"
    ;;
  FULL|RSGP)
    NAME="rsgp_full"
    ;;
  NO_PPN)
    RSGP_USE_PPN_COMPLETION=0
    NAME="rsgp_no_ppn_completion"
    ;;
  NO_RS)
    RSGP_RS_POOL_TOPK=0
    RSGP_USE_GEOMETRY=0
    RSGP_USE_ANCHOR=0
    RSGP_USE_TOPOLOGY=0
    RSGP_USE_TAIL_PRIOR=0
    NAME="rsgp_no_rs_priors"
    ;;
  NO_DEGREE)
    RSGP_USE_DEGREE_SCORE=0
    RSGP_ENFORCE_DEGREE_CAP=0
    NAME="rsgp_no_degree_control"
    ;;
  NO_QUOTA)
    RSGP_ENFORCE_LABEL_QUOTA=0
    NAME="rsgp_no_label_pair_quota"
    ;;
  NO_TAIL)
    RSGP_USE_TAIL_PRIOR=0
    NAME="rsgp_no_tail_prior"
    ;;
  *)
    echo "Usage: bash scripts/eval_predcls_rsgp_ablation.sh \\" >&2
    echo "  PPG|PPN|FULL|NO_PPN|NO_RS|NO_DEGREE|NO_QUOTA|NO_TAIL" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
CHECKPOINT="${CHECKPOINT:-outputs/star_predcls_obb_tail_aux/best_bgfirst.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_ablation_rsgp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${NAME}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINT}" \
CONFIG=configs/star_predcls_obb_tail_aux_train.py \
FILTER_METHOD="${FILTER_METHOD}" \
RSGP_MODE="${RSGP_MODE}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}" \
RSGP_USE_PPN_COMPLETION="${RSGP_USE_PPN_COMPLETION}" \
RSGP_USE_GEOMETRY="${RSGP_USE_GEOMETRY}" \
RSGP_USE_ANCHOR="${RSGP_USE_ANCHOR}" \
RSGP_USE_TOPOLOGY="${RSGP_USE_TOPOLOGY}" \
RSGP_USE_TAIL_PRIOR="${RSGP_USE_TAIL_PRIOR}" \
RSGP_USE_DEGREE_SCORE="${RSGP_USE_DEGREE_SCORE}" \
RSGP_ENFORCE_DEGREE_CAP="${RSGP_ENFORCE_DEGREE_CAP}" \
RSGP_ENFORCE_LABEL_QUOTA="${RSGP_ENFORCE_LABEL_QUOTA}" \
RSGP_RS_POOL_TOPK="${RSGP_RS_POOL_TOPK}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/eval_star_predcls.sh"

