"""STAR OBB sgdet route compatible with SGG-Toolkit's LOBB RPCM command.

The detector is initialized only from ``pretrained/OBB_swin_L_OBD.pth`` and
kept frozen.  Its d1/d2 5-level RPN and oriented RoI branches create post-NMS
detections; the relation head is the sole trainable model component.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)
cfg["MODEL"]["TASK"] = "sgdet"
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_BOX"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_OBJECT_LABEL"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["REQUIRE_BOX_OVERLAP"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["BATCH_SIZE_PER_IMAGE"] = 512
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_FEAT_UPDATE_STEP"] = 3
cfg["feat_update_step"] = 3

# Faithful Sgdets data contract: the frozen d1/d2 patch detector receives the
# original-resolution image while the relation head receives the normal 1024
# relation view.  Disable random geometry until a shared dual-view augmentor
# is deliberately introduced; otherwise raw detector boxes and relation GT
# would no longer share coordinates after rescaling.
for _split in ("TRAIN", "VAL", "TEST"):
    cfg["DATASETS"][_split]["SGDET_DETECTOR_ORIGINAL_RESOLUTION"] = True
cfg["DATASETS"]["TRAIN"]["AUGMENT"] = False

# Original STAR_obb_sgdet.py detector values.
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
        "RPN_NMS_PRE": int(os.environ.get("RPN_NMS_PRE", "2000")),
        "RPN_MAX_PER_IMG": int(os.environ.get("RPN_MAX_PER_IMG", "2000")),
        "RPN_NMS_THRESH": float(os.environ.get("RPN_NMS_THRESH", "0.8")),
        "RCNN_SCORE_THRESH": float(os.environ.get("RCNN_SCORE_THRESH", "0.05")),
        "RCNN_NMS_THRESH": float(os.environ.get("RCNN_NMS_THRESH", "0.1")),
        "RCNN_MAX_PER_IMG": int(os.environ.get("RCNN_MAX_PER_IMG", "2000")),
        "PATCH_MERGE_NMS_THRESH": float(
            os.environ.get("PATCH_MERGE_NMS_THRESH", "0.4")
        ),
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
        "ADD_GTBOX_TO_PROPOSAL_IN_TRAIN": os.environ.get("ADD_GTBOX_TO_PROPOSAL_IN_TRAIN", "1") != "0",
    }
)

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

# LOBB_RPCM_sgdet_train.sh: batch=2, max_iter=5000, milestones=(3000,4000).
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE", "4"))
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = int(os.environ.get("VAL_BATCH_SIZE", "1"))
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE", "1"))
# Full STAR images ride in metadata until the detector crops a patch.  Avoid
# worker-process shared-memory copies of 6k--10kpx float tensors by default.
cfg["DATALOADER"]["NUM_WORKERS"] = int(os.environ.get("NUM_WORKERS", "8"))
cfg["SOLVER"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TRAIN_BATCH_SIZE"]
cfg["TEST"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TEST_BATCH_SIZE"]
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "1e-3"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = os.environ.get("LR_SCALE_BY_BATCH", "1") != "0"
cfg["SOLVER"]["MAX_ITER"] = int(os.environ.get("MAX_ITER", "20000"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "100000"))
cfg["SOLVER"]["STEPS"] = [10000, 14000, 18000]
cfg["SOLVER"]["ITERATION_COMPAT"] = True
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "1000"))
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "10000"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "0"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get("OUTPUT_DIR", "outputs/star_sgdet_obb_train")
