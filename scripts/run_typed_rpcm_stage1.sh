#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE=stage1 exec bash "${SCRIPT_DIR}/run_typed_rpcm_stage.sh" "$@"
