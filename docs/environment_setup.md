# Clean environment for STAR SGG

## Dependency boundary

The training and evaluation paths directly require eight external runtime
packages:

```text
torch, torchvision, numpy, opencv-python, Pillow, h5py, tqdm, mmcv-full
```

These packages are also present in the original RPCM environment, but they are
normal shared framework dependencies. `mmcv-full` is the only unavoidable
OpenMMLab binary dependency because the project uses its rotated IoU, rotated
NMS, and rotated RoIAlign CUDA operators.

The project does **not** import `maskrcnn_benchmark`, `mmdet`, `mmrotate`,
`torch_geometric`, `torch_scatter`, or `torch_sparse`. OBB/polygon conversion is
implemented locally in `sgg/modeling/core/obb_ops.py`; it is numerically aligned
with the three conversion helpers previously imported from mmrotate.

The old `pyg` environment is not isolated. Its `easy-install.pth` contains
absolute paths to:

```text
/home/ubuntu/research/ssd/RPCM
/home/ubuntu/research/ssd/RPCM/mmrote_RS
```

Consequently, importing `maskrcnn_benchmark` or `mmrotate` executes source code
from the original RPCM checkout rather than a self-contained installed wheel.

## Setup from scratch

The installer creates a new minimal environment from pinned dependencies. It
does not read or clone the existing `pyg` environment:

```bash
ENV_NAME=sgg CUDA_HOME=/usr/local/cuda-12.1 MAX_JOBS=8 \
  bash scripts/create_clean_env.sh
conda activate sgg
```

The script installs the tested versions:

```text
Python       3.11
PyTorch      2.2.2+cu121
torchvision  0.17.2+cu121
mmcv-full    1.7.2 (compiled with CUDA ops)
numpy        1.26.4
```

`mmcv-full` 1.7.2 has no generally usable prebuilt wheel for every
Python/PyTorch/CUDA combination. The scratch route therefore compiles it. If a
matching wheel has already been built, avoid recompilation with:

```bash
MMCV_WHEEL=/path/to/mmcv_full-1.7.2-...whl \
ENV_NAME=sgg \
  bash scripts/create_clean_env.sh
```

Do not install both `mmcv` and `mmcv-full`: the packages share the same Python
module name, while only `mmcv-full` contains the required compiled operators.

## Validation

Run the strict environment audit after any dependency change:

```bash
python tools/check_environment.py --strict --require-cuda
```

It verifies pinned versions, executes rotated MMCV operators on CPU and CUDA,
checks the local OBB helpers, rejects absolute RPCM/SGG-ToolKit paths, and
reports any legacy packages still importable. Omit `--require-cuda` only when
checking a CPU-only development shell.

Optional curve plotting and tests use:

```bash
pip install -r requirements.analysis.txt
```
