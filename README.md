# STAR OBB Scene Graph Generation

PyTorch implementation of oriented-bounding-box (OBB) scene graph generation
on STAR. The current project covers Predicate Classification (PredCls), Scene
Graph Classification (SGCls), and Scene Graph Detection (SGDet).

The main method contains three parts:

- **Role-aware Relation Context Aggregation (RCA)** separates
  shared-subject and shared-object edge propagation.
- **Hard-Predicate Residual Calibration (HPRC)** combines weak logit
  adjustment with a zero-initialized residual calibration head. Historical
  code/configuration keys retain the `tail_aux` name for checkpoint
  compatibility.
- **Remote-sensing Graph-aware Pair Proposal (RSGP)** combines PPG, PPN, OBB
  geometry, remote-sensing topology, degree control, and label-pair quotas to
  construct the inference candidate graph.

SGDet uses a read-only detection cache so the frozen full-resolution
multi-scale detector is not rerun during every relation-head update.

## Task protocols

| Task | Boxes | Object labels | Predicates |
|---|---|---|---|
| PredCls | ground truth | ground truth | predicted |
| SGCls | ground truth | predicted | predicted |
| SGDet | predicted | predicted | predicted |

The default SGCls/SGDet pair-filter label sources reproduce the STAR
SGG-ToolKit comparison protocol:

```text
SGCLS_FILTER_LABEL_SOURCE=gt
SGDET_FILTER_LABEL_SOURCE=matched_gt
```

These labels are used for relation candidate filtering only. Final object and
triplet predictions still come from the model. Use `pred` for stricter
fully-predicted-label ablations.

## Repository layout

```text
configs/   OBB task and paper-ablation configurations
scripts/   one-command training, evaluation, cache, and experiment launchers
sgg/       datasets, detector, relation models, structures, and evaluation
tools/     focused diagnostics, cache builders, migrations, and PPN utilities
tests/     numerical, checkpoint-compatibility, and integration tests
train.py   common training entry point
```

Generated datasets, embeddings, detection caches, logs, and checkpoints are
excluded from Git.

## Installation

The tested environment uses Python 3.11, PyTorch 2.2.2 with CUDA 12.1,
torchvision 0.17.2, and `mmcv-full==1.7.2`.

```bash
ENV_NAME=sgg \
CUDA_HOME=/usr/local/cuda-12.1 \
MAX_JOBS=8 \
  bash scripts/create_clean_env.sh

conda activate sgg
python tools/check_environment.py --strict --require-cuda
```

The installer creates a new environment from scratch and does not import the
original RPCM or SGG-ToolKit checkout. See [INSTALL.md](INSTALL.md) for the
complete procedure and [docs/environment_setup.md](docs/environment_setup.md)
for the dependency boundary.

## Data and pretrained artifacts

Set the dataset root:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
```

Expected STAR files:

```text
$STAR_SGG_ROOT/
├── STAR_img/
├── STAR-SGG-with-attri.h5
├── STAR-SGG-dicts-with-attri.json
└── STAR_image_data_v1.json
```

Expected local artifacts:

```text
pretrained/
├── OBB_swin_L_OBD.pth       # frozen OBB detector
├── STAR_OBB.pth             # original PPG
├── PPN_OBB.pth              # independent PPN
└── SF_list_support.json     # semantic label-pair support

glove/
├── glove.6B.200d.txt
└── glove.6B.300d.txt
```

Binary weights and GloVe files are intentionally ignored by Git. The semantic
support JSON remains trackable.

## Training

All commands run in the background and write `train.log`, `train.pid`, and
`exit_code.txt` below their output directory.

### PredCls RPCM/RCA base

```bash
bash scripts/run_star_experiment.sh
```

The default fresh run loads only `pretrained/OBB_swin_L_OBD.pth`; the relation
stack is initialized from its configured RPCM/GloVe initializers.

### PredCls HPRC

```bash
bash scripts/run_star_tail_aux.sh
```

Despite its historical filename, this is the clean HPRC scratch launcher. It
forces `INIT_RPCM=''`, uses the corrected exact-6850 dual-view/GloVe
initialization, and writes to:

```text
outputs/star_predcls_obb_hprc_scratch/
```

The launcher isolates its schedule from generic variables exported by other
experiments. Use HPRC-specific overrides when needed:

```bash
HPRC_MAX_EPOCHS=300 \
HPRC_STEPS=10000,14000,16000 \
HPRC_VAL_START_PERIOD=120 \
bash scripts/run_star_tail_aux.sh
```

### SGCls

```bash
bash scripts/run_star_sgcls_experiment.sh
```

### SGDet

Build the frozen-detector cache once:

```bash
SPLITS=train,test bash scripts/build_sgdet_detection_cache.sh
```

Then train the relation stack from the cache:

```bash
bash scripts/run_star_sgdet_experiment.sh
```

The launcher requires cache hits by default. Changing detector/RPN/RCNN/NMS
settings changes the cache hash and requires rebuilding the cache.

### Resume

Use the same task launcher with a full checkpoint:

```bash
RESUME=outputs/star_predcls_obb_hprc_scratch/model_last.pth \
  bash scripts/run_star_tail_aux.sh

RESUME=outputs/star_sgcls_obb_train/model_last.pth \
  bash scripts/run_star_sgcls_experiment.sh

RESUME=outputs/star_sgdet_obb_train/model_last.pth \
  bash scripts/run_star_sgdet_experiment.sh
```

## Evaluation

Task-specific wrappers automatically choose the corresponding best/last
checkpoint when possible:

```bash
bash scripts/eval_star_predcls.sh
bash scripts/eval_star_sgcls.sh
bash scripts/eval_star_sgdet.sh
```

Choose the pair filter explicitly when comparing methods:

```bash
FILTER_METHOD=PPG bash scripts/eval_star_predcls.sh
FILTER_METHOD=PPN bash scripts/eval_star_predcls.sh
FILTER_METHOD=RSGP bash scripts/eval_star_predcls.sh
```

Use `CHECKPOINT`, `CONFIG`, `OUTPUT_DIR`, and `SPLIT` to override wrapper
defaults. `scripts/eval_once.sh` is the low-level strict entry point used by
all three wrappers.

## Paper experiments

The compact reproducibility entry points are:

```bash
# Existing/scratch PredCls relation-head rows
bash scripts/eval_predcls_minimal_ablation.sh 6850
bash scripts/run_predcls_minimal_ablation.sh A
bash scripts/run_predcls_minimal_ablation.sh B
bash scripts/run_predcls_minimal_ablation.sh C

# Fixed-checkpoint RSGP component rows
bash scripts/eval_predcls_rsgp_ablation.sh FULL
bash scripts/eval_predcls_rsgp_ablation.sh NO_PPN
bash scripts/eval_predcls_rsgp_ablation.sh NO_RS
bash scripts/eval_predcls_rsgp_ablation.sh NO_DEGREE
bash scripts/eval_predcls_rsgp_ablation.sh NO_QUOTA
bash scripts/eval_predcls_rsgp_ablation.sh NO_TAIL

# Cross-task fixed-checkpoint comparisons
bash scripts/eval_cross_task_minimal.sh SGCLS_PPG
bash scripts/eval_cross_task_minimal.sh SGCLS_RSGP
bash scripts/eval_cross_task_minimal.sh SGDET_PPG
bash scripts/eval_cross_task_minimal.sh SGDET_RSGP
```

Method definitions, formulas, current results, table provenance, and remaining
TBD rows are recorded in
[docs/paper_contributions_and_experiments.md](docs/paper_contributions_and_experiments.md).
The RSGP design boundary is recorded in
[docs/rsgp_technical_route.md](docs/rsgp_technical_route.md).

## Diagnostics and tests

```bash
# Environment and rotated CUDA operators
python tools/check_environment.py --strict --require-cuda

# Object/detection error decomposition
bash scripts/diagnose_star_object_classification.sh

# Parse and plot a training log
python scripts/plot_train_log_curves.py outputs/.../train.log

# Optional test/plot dependencies, then the test suite
pip install -r requirements.analysis.txt
pytest -q
```

The complete clean-environment smoke evaluation is:

```bash
bash scripts/smoke_test_clean_env.sh
```

## Reproducibility notes

- `FILTER_METHOD` must be one of `PPG`, `PPN`, or `RSGP`; unfiltered STAR
  relation graphs are disabled because all-pairs inference can cause OOM.
- PredCls HPRC excludes the constant GT object-refinement loss. SGCls and
  SGDet explicitly retain trainable object-refinement CE.
- `outputs/` is not versioned. Preserve checkpoint hashes, configs, logs, and
  metric JSONs used for publication separately.
- Existing historical HPRC results were warm-started from `6850_4135.pth`.
  The current clean HPRC launcher is detector-only scratch training; do not
  conflate the two training protocols.

## License

No license has been selected. Add a `LICENSE` file before public distribution.
