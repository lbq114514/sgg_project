#!/usr/bin/env bash
set -euo pipefail

# Launch exactly one of the three missing PredCls training runs from the
# minimal paper ablation.  Each run is started in the background by
# run_star_experiment.sh; do not start A/B/C together on one GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE="${1:-}"

case "${CASE^^}" in
  A|UNIFIED)
    CONFIG="configs/star_predcls_obb_ablation_unified_rca_train.py"
    DEFAULT_OUTPUT="outputs/star_predcls_ablation_sgg_toolkit_gnn_scratch"
    LABEL="A: SGG-ToolKit GNN + CE (detector-only initialization)"
    ;;
  B|DUAL)
    CONFIG="configs/star_predcls_obb_ablation_dual_rca_train.py"
    DEFAULT_OUTPUT="outputs/star_predcls_ablation_dual_rca_scratch"
    LABEL="B: dual-view RCA + CE (detector-only initialization)"
    ;;
  C|DUAL_LA|LA)
    CONFIG="configs/star_predcls_obb_ablation_dual_la_train.py"
    DEFAULT_OUTPUT="outputs/star_predcls_ablation_dual_la_scratch"
    LABEL="C: dual-view RCA + CE/logit-adjust auxiliary (detector-only initialization)"
    ;;
  *)
    echo "Usage: bash scripts/run_predcls_minimal_ablation.sh A|B|C" >&2
    echo "  A: original SGG-ToolKit heterogeneous GNN, CE" >&2
    echo "  B: role-aware dual-view relation graph, CE" >&2
    echo "  C: role-aware dual-view relation graph, CE + logit adjustment" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT}}"
# A/B/C deliberately initialize only the frozen detector configured by each
# experiment (pretrained/OBB_swin_L_OBD.pth). Do not inherit INIT_RPCM from the
# caller: relation-head and GNN parameters must start identically from scratch.
INIT_RPCM=""
VAL_SPLIT="${VAL_SPLIT:-test}"

echo "Launching ${LABEL}"
echo "Selection split: ${VAL_SPLIT}"
echo "Initialization: detector-only (pretrained/OBB_swin_L_OBD.pth)"

CONFIG="${CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
INIT_RPCM="${INIT_RPCM}" \
FILTER_METHOD=PPG \
VAL_SPLIT="${VAL_SPLIT}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"
