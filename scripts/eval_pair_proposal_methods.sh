#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pair_proposal_comparison}"
PPN_A_CHECKPOINT="${PPN_A_CHECKPOINT:-outputs/star_pair_proposal_network_smoke/model_best.pth}"
PPN_B_CHECKPOINT="${PPN_B_CHECKPOINT:-outputs/star_pair_proposal_network_v2/model_best.pth}"
PPG_CHECKPOINT="${PPG_CHECKPOINT:-pretrained/STAR_OBB.pth}"
BLOCK_SIZE="${BLOCK_SIZE:-200000}"
mkdir -p "${OUTPUT_DIR}"

common=(--device "${DEVICE}" --split "${SPLIT}" --topk 1000 2000 5000 10000 --block-size "${BLOCK_SIZE}")

python tools/eval_pair_proposal_network.py "${common[@]}" \
  --method PPN --run-name PPN_A --checkpoint "${PPN_A_CHECKPOINT}" --output "${OUTPUT_DIR}/ppn_a.json"
python tools/eval_pair_proposal_network.py "${common[@]}" \
  --method PPN --run-name PPN_B --checkpoint "${PPN_B_CHECKPOINT}" --output "${OUTPUT_DIR}/ppn_b.json"
python tools/eval_pair_proposal_network.py "${common[@]}" \
  --method PPG --ppg-checkpoint "${PPG_CHECKPOINT}" --output "${OUTPUT_DIR}/ppg.json"

python tools/compare_pair_proposal_metrics.py \
  "${OUTPUT_DIR}/ppn_a.json" "${OUTPUT_DIR}/ppn_b.json" "${OUTPUT_DIR}/ppg.json" \
  --output "${OUTPUT_DIR}/comparison.json"
