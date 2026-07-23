#!/usr/bin/env bash
set -euo pipefail

# Build a new minimal environment without reading or cloning the RPCM/pyg
# environment. PyTorch is installed for CUDA 12.1 and mmcv-full is compiled
# against that exact PyTorch installation unless MMCV_WHEEL is provided.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sgg}"
CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
MAX_JOBS="${MAX_JOBS:-8}"
EXPECTED_PYTHON="3.11"
RESUME_INSTALL="${RESUME_INSTALL:-0}"

if [[ -z "${CONDA_BIN}" ]]; then
  echo "conda was not found. Activate a Conda installation first." >&2
  exit 1
fi

CONFIGURED_PYTHON="$(sed -n 's/^[[:space:]]*- python=\([0-9][0-9.]*\).*/\1/p' "${ROOT_DIR}/environment.clean.yml" | head -n 1)"
if [[ "${CONFIGURED_PYTHON}" != "${EXPECTED_PYTHON}" ]]; then
  echo "Unsupported Python version in environment.clean.yml: ${CONFIGURED_PYTHON:-unknown}" >&2
  echo "This stack requires Python ${EXPECTED_PYTHON} for torch 2.2.2 and mmcv-full 1.7.2." >&2
  exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  if [[ "${RESUME_INSTALL}" != "1" ]]; then
    echo "Conda environment already exists: ${ENV_NAME}" >&2
    echo "Set RESUME_INSTALL=1 to continue an interrupted installation." >&2
    echo "Otherwise remove it explicitly or choose another ENV_NAME." >&2
    exit 1
  fi
  echo "Continuing installation in existing environment: ${ENV_NAME}"
else
  "${CONDA_BIN}" env create -y --name "${ENV_NAME}" --file "${ROOT_DIR}/environment.clean.yml"
fi

ENV_PREFIX="$("${CONDA_BIN}" env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF}')"
PYTHON="${ENV_PREFIX}/bin/python"

# mmcv-full 1.7.2 imports pkg_resources during metadata generation. Modern
# setuptools releases have removed that module, so use the legacy build tools
# that are known to work with this source package. Pin pip as well to keep its
# PEP 517 behavior stable across future Conda repository updates.
"${PYTHON}" -m pip install --upgrade --force-reinstall \
  pip==24.0 setuptools==60.2.0 wheel==0.45.1
"${PYTHON}" -c "import pkg_resources; print('Legacy MMCV build backend:', pkg_resources.__file__)"

"${PYTHON}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

"${PYTHON}" -m pip install -r "${ROOT_DIR}/requirements.clean.txt"

if [[ -n "${MMCV_WHEEL:-}" ]]; then
  "${PYTHON}" -m pip install "${MMCV_WHEEL}"
else
  echo "Building mmcv-full 1.7.2 against PyTorch/CUDA; this may take several minutes."
  MMCV_WITH_OPS=1 CUDA_HOME="${CUDA_HOME}" MAX_JOBS="${MAX_JOBS}" \
    "${PYTHON}" -m pip install --no-build-isolation --verbose mmcv-full==1.7.2
fi

"${PYTHON}" "${ROOT_DIR}/tools/check_environment.py" --strict --require-cuda

echo
echo "Environment is ready: ${ENV_NAME}"
echo "Activate it with: conda activate ${ENV_NAME}"
