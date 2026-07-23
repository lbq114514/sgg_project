#!/usr/bin/env bash
set -euo pipefail

# Evaluate one row of the minimal PredCls relation-head ablation.  The wrapper
# starts evaluation in the background and writes test.log/test_metrics.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE="${1:-}"

case "${CASE^^}" in
  A|UNIFIED)
    CONFIG="configs/star_predcls_obb_ablation_unified_rca_train.py"
    DEFAULT_CHECKPOINT="outputs/star_predcls_ablation_sgg_toolkit_gnn_scratch/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_ablation_predcls/A_sgg_toolkit_gnn_ppg"
    FILTER_METHOD="PPG"
    ;;
  B|DUAL)
    CONFIG="configs/star_predcls_obb_ablation_dual_rca_train.py"
    DEFAULT_CHECKPOINT="outputs/star_predcls_ablation_dual_rca_scratch/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_ablation_predcls/B_dual_rca_ppg"
    FILTER_METHOD="PPG"
    ;;
  BASE|6850|RPCM_6850)
    CONFIG="configs/star_predcls_obb_ablation_dual_rca_train.py"
    DEFAULT_CHECKPOINT="/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth"
    DEFAULT_OUTPUT="outputs/paper_ablation_predcls/B_dual_rca_6850_ppg"
    DEFAULT_LOAD_MODE="legacy-rpcm"
    FILTER_METHOD="PPG"
    ;;
  C|DUAL_LA|LA)
    CONFIG="configs/star_predcls_obb_ablation_dual_la_train.py"
    DEFAULT_CHECKPOINT="outputs/star_predcls_ablation_dual_la_scratch/model_best_HR.pth"
    DEFAULT_OUTPUT="outputs/paper_ablation_predcls/C_dual_la_ppg"
    FILTER_METHOD="PPG"
    ;;
  D|FULL_PPG)
    CONFIG="configs/star_predcls_obb_tail_aux_train.py"
    DEFAULT_CHECKPOINT="outputs/star_predcls_obb_tail_aux/best_bgfirst.pth"
    DEFAULT_OUTPUT="outputs/paper_ablation_predcls/D_role_rca_hprc_ppg"
    FILTER_METHOD="PPG"
    ;;
  E|FULL_RSGP)
    CONFIG="configs/star_predcls_obb_tail_aux_train.py"
    DEFAULT_CHECKPOINT="outputs/star_predcls_obb_tail_aux/best_bgfirst.pth"
    DEFAULT_OUTPUT="outputs/paper_ablation_predcls/E_role_rca_hprc_rsgp"
    FILTER_METHOD="RSGP"
    ;;
  *)
    echo "Usage: bash scripts/eval_predcls_minimal_ablation.sh 6850|A|B|C|D|E" >&2
    echo "  6850: historical dual-view base; A-C: optional scratch runs; D-E: existing HPRC checkpoint" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT}}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-${DEFAULT_LOAD_MODE:-full}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

CONFIG="${CONFIG}" \
CHECKPOINT="${CHECKPOINT}" \
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE}" \
FILTER_METHOD="${FILTER_METHOD}" \
RSGP_MODE=HYBRID \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/eval_star_predcls.sh"
