#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sgg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_tail_aux_train.py}"
CHECKPOINT="${CHECKPOINT:-outputs/star_predcls_obb_tail_aux/best.pth}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-full}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rsgp_grid}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
PPG_MODEL_PATH_OBB="${PPG_MODEL_PATH_OBB:-pretrained/STAR_OBB.pth}"
PPN_MODEL_PATH="${PPN_MODEL_PATH:-pretrained/PPN_OBB.pth}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

run_case() {
  local name="$1"
  local filter_method="$2"
  local rsgp_mode="${3:-HYBRID}"
  local protected_topk="${4:-7000}"

  local out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"

  echo "Running ${name}: filter=${filter_method}, rsgp_mode=${rsgp_mode}, protected=${protected_topk}"
  env \
    FILTER_METHOD="${filter_method}" \
    TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    PPG_MODEL_PATH_OBB="${PPG_MODEL_PATH_OBB}" \
    PPN_MODEL_PATH="${PPN_MODEL_PATH}" \
    RSGP_MODE="${rsgp_mode}" \
    RSGP_PPG_PROTECTED_TOPK="${protected_topk}" \
    python tools/eval_once.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --checkpoint-load-mode "${CHECKPOINT_LOAD_MODE}" \
      --split "${SPLIT}" \
      --device "${DEVICE}" \
      --filter-method "${filter_method}" \
      --output "${out_dir}/test_metrics.json" \
      > "${out_dir}/test.log" 2>&1
}

run_case "ppg_10000" "PPG"
run_case "ppn_10000" "PPN"
run_case "rsgp_rs_only" "RSGP" "RS_ONLY" "0"
run_case "rsgp_ppn_graph" "RSGP" "PPN_GRAPH" "0"
run_case "rsgp_hybrid_9000_1000" "RSGP" "HYBRID" "9000"
run_case "rsgp_hybrid_8000_2000" "RSGP" "HYBRID" "8000"
run_case "rsgp_hybrid_7000_3000" "RSGP" "HYBRID" "7000"

python - "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("*/test_metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    coverage = metrics.get("candidate-stage-coverage", {})
    rows.append(
        {
            "case": metrics_path.parent.name,
            "R@1500": metrics.get("R", {}).get("1500"),
            "R@2000": metrics.get("R", {}).get("2000"),
            "mR@1500": metrics.get("mR", {}).get("1500"),
            "mR@2000": metrics.get("mR", {}).get("2000"),
            "HMR@1500": metrics.get("HR", {}).get("1500"),
            "HMR@2000": metrics.get("HR", {}).get("2000"),
            "final_pair_coverage": coverage.get("final"),
            "ppn_pool_coverage": coverage.get("ppn"),
            "strict_degree_cap_coverage": coverage.get("degree_cap"),
        }
    )
out = root / "comparison.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY

echo "Done. Logs and JSON files are under ${OUTPUT_ROOT}"
