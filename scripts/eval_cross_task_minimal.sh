#!/usr/bin/env bash
set -euo pipefail

# Cross-task generalization check. It never retrains SGCls/SGDet: PPG and RSGP
# are evaluated with the exact same task checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE="${1:-}"

case "${CASE^^}" in
  SGCLS_PPG)
    TASK=sgcls
    FILTER_METHOD=PPG
    DEFAULT_CHECKPOINT="outputs/star_sgcls_obb_train/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_cross_task/sgcls_ppg"
    ;;
  SGCLS_RSGP)
    TASK=sgcls
    FILTER_METHOD=RSGP
    DEFAULT_CHECKPOINT="outputs/star_sgcls_obb_train/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_cross_task/sgcls_rsgp"
    ;;
  SGDET_PPG)
    TASK=sgdet
    FILTER_METHOD=PPG
    DEFAULT_CHECKPOINT="outputs/star_sgdet_obb_train/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_cross_task/sgdet_ppg"
    ;;
  SGDET_RSGP)
    TASK=sgdet
    FILTER_METHOD=RSGP
    DEFAULT_CHECKPOINT="outputs/star_sgdet_obb_train/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_cross_task/sgdet_rsgp"
    ;;
  *)
    echo "Usage: bash scripts/eval_cross_task_minimal.sh \\" >&2
    echo "  SGCLS_PPG|SGCLS_RSGP|SGDET_PPG|SGDET_RSGP" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${TASK}" == "sgcls" ]]; then
  CHECKPOINT="${CHECKPOINT}" \
  FILTER_METHOD="${FILTER_METHOD}" \
  RSGP_MODE=HYBRID \
  RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}" \
  SGCLS_FILTER_LABEL_SOURCE=gt \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash "${SCRIPT_DIR}/eval_star_sgcls.sh"
else
  CHECKPOINT="${CHECKPOINT}" \
  FILTER_METHOD="${FILTER_METHOD}" \
  RSGP_MODE=HYBRID \
  RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}" \
  SGDET_FILTER_LABEL_SOURCE=matched_gt \
  SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}" \
  SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-outputs/star_sgdet_detection_cache}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash "${SCRIPT_DIR}/eval_star_sgdet.sh"
fi
