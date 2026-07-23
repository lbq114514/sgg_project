# Environment Installation

This project uses a small runtime stack for STAR scene graph generation. It
does not require the original RPCM checkout, `maskrcnn_benchmark`, mmrotate,
mmdet, or the PyTorch Geometric packages.

## Tested stack

```text
Python       3.11
PyTorch      2.2.2+cu121
torchvision  0.17.2+cu121
CUDA toolkit 12.1
mmcv-full    1.7.2, compiled with CUDA operators
numpy        1.26.4
```

The project directly uses these runtime packages:

```text
torch, torchvision, numpy, opencv-python, Pillow, h5py, tqdm, mmcv-full
```

`mmcv-full` is required for rotated IoU, rotated NMS, and rotated RoIAlign.
Install `mmcv-full`, not the lightweight `mmcv` distribution. Never install
both into the same environment because they provide the same `mmcv` Python
module.

## Prerequisites

- A working Conda installation.
- An NVIDIA driver compatible with CUDA 12.1.
- CUDA 12.1 toolkit at `/usr/local/cuda-12.1` when compiling MMCV locally.
- Sufficient disk space for PyTorch and the compiled MMCV extension.

Check the host before installation:

```bash
nvidia-smi
/usr/local/cuda-12.1/bin/nvcc --version
conda --version
```

## Create a minimal environment from scratch

The installer always creates a new environment from pinned dependencies. It
does not clone, inspect, import, or modify the existing `pyg` environment and
does not use source code from the RPCM checkout.

Run from the project root:

```bash
cd /home/ubuntu/research/ssd/sgg_project

ENV_NAME=sgg \
CUDA_HOME=/usr/local/cuda-12.1 \
MAX_JOBS=8 \
  bash scripts/create_clean_env.sh

conda activate sgg
```

The script creates the Conda environment from `environment.clean.yml`, installs
PyTorch and the pinned Python dependencies, and builds `mmcv-full==1.7.2` with
CUDA operators. The MMCV build can take several minutes and uses substantial
CPU memory. Reduce `MAX_JOBS` if compilation exhausts host memory.

The installer deliberately pins the legacy build backend below the modern
runtime package versions:

```text
pip        24.0
setuptools 60.2.0 (includes pkg_resources)
wheel      0.45.1
```

These pins are required only because `mmcv-full 1.7.2` still imports
`pkg_resources` from its setup script.

If a matching MMCV wheel is available, skip source compilation:

```bash
ENV_NAME=sgg \
MMCV_WHEEL=/absolute/path/to/mmcv_full-1.7.2-...whl \
  bash scripts/create_clean_env.sh
```

The wheel must match all four components:

```text
Python 3.11 + PyTorch 2.2.2 + CUDA 12.1 + Linux x86_64
```

## Validate the installation

The environment creation script performs this check automatically. It can also
be rerun after any dependency change:

```bash
conda activate sgg
cd /home/ubuntu/research/ssd/sgg_project

python tools/check_environment.py --strict --require-cuda
```

The command verifies:

- exact versions of the eight direct runtime packages;
- CPU and CUDA execution of MMCV rotated operators;
- local OBB geometry helpers;
- absence of RPCM or SGG-ToolKit paths in `sys.path`;
- absence of maskrcnn, mmrotate, mmdet, and PyG legacy packages.

For a CPU-only development shell, omit `--require-cuda`:

```bash
python tools/check_environment.py --strict
```

## Optional analysis and testing tools

Training and evaluation do not require matplotlib or pytest. Install them only
when plotting logs or running tests:

```bash
pip install -r requirements.analysis.txt
pytest -q
```

## Quick project smoke test

Run the complete smoke test from the project root after activating the new
environment:

```bash
conda activate sgg
cd /home/ubuntu/research/ssd/sgg_project
bash scripts/smoke_test_clean_env.sh
```

The script first validates CUDA and MMCV, then evaluates the existing PredCls
checkpoint on two real STAR test images with PPG. It covers dataset loading,
detector/RPCM checkpoint loading, OBB operators, pair proposal, relation
inference, and metric output. Successful execution writes:

```text
outputs/environment_smoke_test/test_metrics.json
```

The test size and paths can be overridden without editing the script:

```bash
MAX_IMAGES=4 \
CONFIG=configs/star_predcls_obb_tail_aux_train.py \
CHECKPOINT=outputs/star_predcls_obb_tail_aux/best.pth \
OUTPUT_DIR=outputs/environment_smoke_test_4 \
  bash scripts/smoke_test_clean_env.sh
```

For an import-only check without STAR data or a checkpoint:

```bash
python - <<'PY'
import torch
from mmcv.ops import box_iou_rotated, nms_rotated
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Project and rotated MMCV operators imported successfully")
PY
```

## Common problems

### MMCV metadata generation reports `No module named 'pkg_resources'`

Recent Conda repositories may install setuptools 81 or newer, where
`pkg_resources` is no longer available. The current installer pins
`setuptools==60.2.0` before building MMCV.

If environment creation already reached this failure, continue it without
deleting the downloaded PyTorch packages:

```bash
RESUME_INSTALL=1 \
ENV_NAME=sgg \
CUDA_HOME=/usr/local/cuda-12.1 \
MAX_JOBS=8 \
  bash scripts/create_clean_env.sh
```

The resume path reapplies all version pins and is safe after an interrupted
installation.

### `PackagesNotFoundError` for Ninja after changing Python

Do not change only `python=3.11` to Python 3.12. The exact Ninja build may be
the first dependency that Conda reports as unavailable, but it is not the main
compatibility boundary: the pinned PyTorch 2.2.2 CUDA wheel and the old
`mmcv-full==1.7.2` extension are also tied to the tested Python 3.11 stack.

The environment intentionally leaves Ninja's patch version unconstrained
because it is only a compilation accelerator. Restore `python=3.11` and rerun:

```bash
ENV_NAME=sgg CUDA_HOME=/usr/local/cuda-12.1 \
  bash scripts/create_clean_env.sh
```

Supporting Python 3.12 would require a separate migration of PyTorch,
torchvision, and MMCV followed by numerical and checkpoint regression tests;
it is not a drop-in environment-file change.

### The target environment already exists

The installer refuses to overwrite an existing complete environment unless
`RESUME_INSTALL=1` is supplied. Choose a new name:

```bash
ENV_NAME=sgg-v2 \
  bash scripts/create_clean_env.sh
```

Alternatively, remove the old environment explicitly before recreating it:

```bash
conda env remove -n sgg
```

### `mmcv._ext` cannot be imported

This normally means MMCV was built for a different PyTorch, Python, or CUDA
version. Remove both MMCV distributions and rebuild using the pinned stack:

```bash
pip uninstall -y mmcv mmcv-full

MMCV_WITH_OPS=1 CUDA_HOME=/usr/local/cuda-12.1 MAX_JOBS=8 \
  pip install --no-build-isolation --verbose mmcv-full==1.7.2
```

### `undefined symbol` while importing an MMCV operator

The compiled extension and PyTorch ABI do not match. Do not upgrade PyTorch in
place. Recreate the environment, or install an MMCV wheel built against exactly
PyTorch 2.2.2 and CUDA 12.1.

### RPCM paths appear in the audit

Inspect `.pth` and editable-link files in the active environment:

```bash
python - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    for path in Path(root).glob("*.pth"):
        text = path.read_text(errors="replace")
        if "/RPCM" in text or "/SGG-ToolKit" in text:
            print(path)
            print(text)
PY
```

This should not happen in a newly created environment. It normally means that
`PYTHONPATH` contains a legacy checkout path in the current shell. Clear it and
rerun the audit:

```bash
unset PYTHONPATH
python tools/check_environment.py --strict --require-cuda
```

### CUDA is visible to `nvidia-smi` but not to PyTorch

Confirm that the CUDA-enabled PyTorch build is installed:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

The expected outputs include `2.2.2+cu121`, CUDA `12.1`, and `True`.
