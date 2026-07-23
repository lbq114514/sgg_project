"""Main configuration for STAR OBB Scene Graph Detection (SGDet).

Task protocol: boxes, object labels, and predicates are all predicted. The
frozen detector is initialized from ``pretrained/OBB_swin_L_OBD.pth`` and uses
the source d1/d2 multi-scale patch, RPN, oriented RoI, and cross-patch NMS
pipeline to generate proposals. Only the relation stack is updated in training.

Performance constraint: raw remote-sensing images can exceed 10k pixels. Do
not rerun the frozen detector at every relation step. Formal training reads
the v5 detection cache by default:
    outputs/star_sgdet_detection_cache/{train,test}/*.pt

The cache stores detector proposals/logits only, not GT labels or relations.
The same cache can therefore switch between matched_gt and pred label
protocols. A hash or version mismatch raises an error instead of silently
falling back to the slow detector.

Typical workflow:
    bash scripts/build_sgdet_detection_cache.sh
    bash scripts/run_star_sgdet_experiment.sh
"""

from __future__ import annotations

import copy
import os

# Keep the relation-head structure aligned with the predcls checkpoint used as
# cross-task initialization.  In particular, best_bgfirst.pth contains the
# 15-predicate HPRC head (historical ``tail_aux`` state-dict names);
# inheriting the plain predcls config would silently build an empty calibration
# head and make the checkpoint structurally incompatible.
from configs.star_predcls_obb_tail_aux_train import cfg as _base_cfg


# ---------------------------------------------------------------------------
# 1. SGDet task: predicted boxes + predicted object labels + predicates
# ---------------------------------------------------------------------------
cfg = copy.deepcopy(_base_cfg)
cfg["MODEL"]["TASK"] = "sgdet"
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_BOX"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_OBJECT_LABEL"] = False
# Unlike PredCls, SGDet learns the RPCM object-refinement classifier on
# detector proposals matched according to the selected training protocol.
cfg["MODEL"]["ROI_RELATION_HEAD"]["OBJECT_REFINE_LOSS_WEIGHT"] = 1.0
cfg["MODEL"]["ROI_RELATION_HEAD"]["REQUIRE_BOX_OVERLAP"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["BATCH_SIZE_PER_IMAGE"] = 512
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_FEAT_UPDATE_STEP"] = 3
cfg["feat_update_step"] = 3
# The config itself disables the cache by default so the raw detector can be
# debugged directly. The formal launcher explicitly sets
# SGDET_DETECTION_CACHE_ENABLED=1.
_sgdet_detection_cache_enabled = os.environ.get(
    "SGDET_DETECTION_CACHE_ENABLED", "0"
).strip().lower() not in {"0", "false", "no", "off", ""}

# Faithful Sgdets data contract: the frozen d1/d2 patch detector receives the
# original-resolution image while the relation head receives the normal 1024
# relation view.  Disable random geometry until a shared dual-view augmentor
# is deliberately introduced; otherwise raw detector boxes and relation GT
# would no longer share coordinates after rescaling.
for _split in ("TRAIN", "VAL", "TEST"):
    cfg["DATASETS"][_split]["SGDET_DETECTOR_ORIGINAL_RESOLUTION"] = (
        not _sgdet_detection_cache_enabled
    )
cfg["DATASETS"]["TRAIN"]["AUGMENT"] = False

# ---------------------------------------------------------------------------
# 2. Frozen-detector protocol from STAR_obb_sgdet.py
# RPN/RCNN/NMS settings are included in the cache hash. Rebuild the cache
# after changing any of them.
# ---------------------------------------------------------------------------
cfg["MODEL"]["ROI_HEADS"]["DETECTIONS_PER_IMG"] = int(
    os.environ.get("RCNN_MAX_PER_IMG", "2000")
)
cfg["MODEL"]["SGDET_COMPAT"].update(
    {
        "ENABLED": True,
        "FREEZE_DETECTOR": True,
        "USE_D2": os.environ.get("SGDET_USE_D2", "1") != "0",
        "D2_SCALE": float(os.environ.get("SGDET_D2_SCALE", "0.5")),
        "RPN_ANCHOR_SIZES": [32, 64, 128, 256, 512],
        "RPN_ASPECT_RATIOS": [0.5, 1.0, 2.0],
        "RPN_ANCHOR_OFFSET": float(os.environ.get("RPN_ANCHOR_OFFSET", "0.0")),
        "RPN_NMS_PRE": int(os.environ.get("RPN_NMS_PRE", "2000")),
        "RPN_MAX_PER_IMG": int(os.environ.get("RPN_MAX_PER_IMG", "2000")),
        "RPN_NMS_THRESH": float(os.environ.get("RPN_NMS_THRESH", "0.8")),
        "RCNN_SCORE_THRESH": float(os.environ.get("RCNN_SCORE_THRESH", "0.05")),
        "RCNN_NMS_THRESH": float(os.environ.get("RCNN_NMS_THRESH", "0.1")),
        "RCNN_MAX_PER_IMG": int(os.environ.get("RCNN_MAX_PER_IMG", "2000")),
        "PATCH_MERGE_NMS_THRESH": float(
            os.environ.get("PATCH_MERGE_NMS_THRESH", "0.4")
        ),
        # matched_gt reproduces the source STAR protocol: GT-matched classes
        # are used by relation sampling/filtering. pred strictly uses detector
        # classes. Final sgdet metrics still use model outputs, and choosing
        # matched_gt here never injects GT boxes into test proposals.
        "TRAIN_LABEL_SOURCE": os.environ.get("SGDET_TRAIN_LABEL_SOURCE", "matched_gt"),
        # Match SGG-Toolkit/STAR's legacy sgdet protocol by default: detector
        # boxes keep detector logits/pred_labels, while proposal.labels is set
        # from detector-to-GT matching for pair filtering and relation sampling.
        # Use SGDET_FILTER_LABEL_SOURCE=pred for strict predicted-label
        # ablations.
        "EVAL_LABEL_SOURCE": os.environ.get(
            "SGDET_FILTER_LABEL_SOURCE",
            os.environ.get("SGDET_EVAL_LABEL_SOURCE", "matched_gt"),
        ),
        "PATCH_DEBUG": os.environ.get("SGDET_PATCH_DEBUG", "0").strip().lower()
        not in {"0", "false", "no", "off", ""},
        # The source STAR Sgdets route runs the frozen detector via
        # ``batch() -> simple_test()`` during relation training.  GT boxes and
        # relations supervise matching/sampling, but GT boxes are not appended
        # to the post-NMS detector proposal set.  Set the environment variable
        # to 1 only for an explicit GT-injection ablation.
        "ADD_GTBOX_TO_PROPOSAL_IN_TRAIN": os.environ.get(
            "ADD_GTBOX_TO_PROPOSAL_IN_TRAIN", "0"
        )
        != "0",
        # REQUIRE_HIT=True is the safe formal-training setting. A missing file
        # or hash mismatch stops immediately instead of making one batch fall
        # back to the multi-minute raw detector.
        "DETECTION_CACHE": {
            "ENABLED": _sgdet_detection_cache_enabled,
            "DIR": os.environ.get(
                "SGDET_DETECTION_CACHE_DIR",
                "outputs/star_sgdet_detection_cache",
            ),
            "REQUIRE_HIT": os.environ.get("SGDET_DETECTION_CACHE_REQUIRE_HIT", "1").strip().lower()
            not in {"0", "false", "no", "off", ""},
            "HASH": os.environ.get("SGDET_DETECTION_CACHE_HASH", ""),
        },
    }
)

# ---------------------------------------------------------------------------
# 3. Pair filter: prunes only the relation graph, not detector proposals
# FILTER_METHOD is mandatory. SGDet can contain many proposals, so an
# all-pairs graph has quadratic memory cost.
# ---------------------------------------------------------------------------
# FILTER_METHOD is mandatory: the relation graph is never allowed to use all
# detection pairs. PPG is the first sgdet baseline; RSGP can be selected after
# this route is validated.
_filter_method = os.environ.get("FILTER_METHOD", "RSGP").upper()
if _filter_method not in {"PPG", "PPN", "RSGP"}:
    raise ValueError("FILTER_METHOD must be PPG, PPN, or RSGP; unfiltered pair graphs are disabled.")
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"] = _filter_method
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_ENABLED"] = _filter_method == "PPG"
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_ENABLED"] = _filter_method == "PPN"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_ENABLED"] = _filter_method == "RSGP"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_MODE"] = os.environ.get("RSGP_MODE", "HYBRID").upper()
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_TOPK"] = int(os.environ.get("RSGP_TOPK", "10000"))
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_PPG_PROTECTED_TOPK"] = int(
    os.environ.get("RSGP_PPG_PROTECTED_TOPK", "8000")
)
# PPG scores candidates before reducing to top-10000.  Bound the temporary
# score/features chunk, not the original detector's post-NMS candidate set.
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_CHUNK_SIZE"] = int(
    os.environ.get("PPG_CHUNK_SIZE", "200000")
)

# ---------------------------------------------------------------------------
# 4. Iteration-based relation training
# MAX_ITER is the effective hard stop; MAX_EPOCHS is only a safety ceiling.
# With ITERATION_COMPAT=True, STEPS, VAL_START_PERIOD, and VAL_PERIOD are
# optimizer-step counts. Cache-backed training can use a larger batch than the
# source raw-detector route.
# ---------------------------------------------------------------------------
# LOBB_RPCM_sgdet_train.sh: batch=2, max_iter=5000, milestones=(3000,4000).
# sgdet runs the frozen full-resolution patch detector before every relation
# update, so each optimizer step is much slower than predcls/sgcls.  Keep the
# default schedule short and make all iteration controls environment
# overridable for resume/debug runs.
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE", "8"))
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = int(os.environ.get("VAL_BATCH_SIZE", "1"))
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE", "1"))
# Full STAR images ride in metadata until the detector crops a patch.  Avoid
# worker-process shared-memory copies of 6k--10kpx float tensors by default.
cfg["DATALOADER"]["NUM_WORKERS"] = int(os.environ.get("NUM_WORKERS", "8"))
cfg["SOLVER"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TRAIN_BATCH_SIZE"]
cfg["TEST"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TEST_BATCH_SIZE"]
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "1e-3"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = os.environ.get("LR_SCALE_BY_BATCH", "1") != "0"
cfg["SOLVER"]["MAX_ITER"] = int(os.environ.get("MAX_ITER", "12000"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "100000"))
_steps = os.environ.get("STEPS", "8000,10000")
cfg["SOLVER"]["STEPS"] = [
    int(item) for item in _steps.replace(";", ",").split(",") if item.strip()
]
cfg["SOLVER"]["ITERATION_COMPAT"] = True
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "200"))
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "7000"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "0"))
cfg["SOLVER"]["PRINT_TRAIN_STEP_FREQ"] = int(os.environ.get("PRINT_TRAIN_STEP_FREQ", "0"))
cfg["SOLVER"]["PRINT_TRAIN_BATCH_FREQ"] = int(os.environ.get("PRINT_TRAIN_BATCH_FREQ", "0"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get("OUTPUT_DIR", "outputs/star_sgdet_obb_train")
