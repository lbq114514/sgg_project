# STAR Scene Graph Generation

PyTorch research code for scene graph generation (SGG) with horizontal bounding boxes (HBB) and oriented bounding boxes (OBB). The repository contains a modular detector, ROI and relation heads, STAR dataset loading, training/evaluation entry points, and ablation configurations.

## Features

- HBB and OBB support through a shared `BoxList` abstraction
- Swin Transformer backbone and feature pyramid network
- Oriented RPN, ROI extraction, and box heads
- Predicate classification with RPCM, prototype learning, PPG, and semantic filtering
- STAR-SGG training, evaluation, analysis, and ablation scripts

## Repository layout

```text
configs/       STAR training and ablation configurations
scripts/       experiment launchers and analysis scripts
sgg/           datasets, models, structures, evaluation, and training engine
tools/         dataset and experiment utilities
toy_data/      small tracked sample dataset
train.py       training entry point
eval.py        evaluation entry point
train_demo.py  synthetic-data smoke test
```

Generated outputs, logs, datasets, and model checkpoints are intentionally excluded from Git. In particular, the multi-GB files under `outputs/` and `pretrained/` must be distributed separately (for example through a release, object storage, or Git LFS).

## Installation

Python 3.10+ is recommended. Install PyTorch for the CUDA version on your machine first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`mmcv` may require a wheel matching the installed PyTorch/CUDA combination. Follow its platform-specific installation instructions if a regular `pip` install cannot build it.

## Data and pretrained files

Set `STAR_SGG_ROOT` to a STAR-SGG directory with this structure:

```text
$STAR_SGG_ROOT/
├── STAR_img/
├── STAR-SGG-with-attri.h5
├── STAR-SGG-dicts-with-attri.json
└── STAR_image_data_v1.json
```

The main configuration also expects these files locally:

```text
pretrained/
├── OBB_swin_L_OBD.pth
├── STAR_HBB.pth
├── STAR_OBB.pth
└── SF_list_support.json
```

The JSON semantic-filter files are small enough to commit; checkpoint files are ignored by default.

## Usage

Run the synthetic smoke test without STAR data:

```bash
python train_demo.py
```

Train on STAR-SGG:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
python train.py --config configs/star_predcls_obb_train.py --device cuda
```

Evaluate a checkpoint:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
python eval.py \
  --config configs/star_predcls_obb_train.py \
  --checkpoint /path/to/model.pth \
  --device cuda
```

The scripts in `scripts/` support background experiment runs, resume workflows, and analysis. Their conda environment and paths can be overridden through environment variables documented at the top of each script.

## Reproducibility notes

- The default STAR configuration disables cuDNN and uses deterministic settings.
- Experiment artifacts are written below `outputs/` and are not versioned.
- Record the exact Python, PyTorch, CUDA, and MMCV versions used for published results.

## License

No license has been selected yet. Add a `LICENSE` file before distributing the project if reuse terms should be granted.
