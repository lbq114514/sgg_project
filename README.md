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

GloVe text embeddings are loaded from `glove/glove.6B.200d.txt`. Pair proposal training uses the converted tensor file `glove/glove.6B.200d.pt`. The local `glove/` directory is excluded from Git because these files are large.

## Usage

Run the synthetic smoke test without STAR data:

```bash
python train_demo.py
```

Train on STAR-SGG:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
bash scripts/run_star_experiment.sh
```

Evaluate a checkpoint:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
bash scripts/eval_star_predcls.sh
```

The active STAR experiment path is the aligned RPCM + PPG predcls setup. The scripts in `scripts/` are intentionally kept small:

- `run_star_experiment.sh`: launch the main aligned RPCM training flow.
- `eval_once.sh`: strict low-level evaluation entry; requires explicit config, checkpoint, and output directory.
- `eval_star_predcls.sh`, `eval_star_sgcls.sh`, `eval_star_sgdet.sh`: task-specific one-command evaluation wrappers.
- `resume_star.sh`: resume the aligned RPCM training flow.

## Reproducibility notes

- The default STAR configuration disables cuDNN and uses deterministic settings.
- Experiment artifacts are written below `outputs/` and are not versioned.
- Record the exact Python, PyTorch, CUDA, and MMCV versions used for published results.

## License

No license has been selected yet. Add a `LICENSE` file before distributing the project if reuse terms should be granted.
