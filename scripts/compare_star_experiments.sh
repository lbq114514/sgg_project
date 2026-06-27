#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
K="${K:-2000}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/compare_star_experiments.py --k "${K}" "$@"
