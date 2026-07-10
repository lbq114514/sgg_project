"""STAR OBB sgcls baseline with selectable pair-filter label protocol."""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)
cfg["MODEL"]["TASK"] = "sgcls"
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_BOX"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_OBJECT_LABEL"] = False
# Match SGG-Toolkit LOBB_RPCM_sgcls_train.sh: the OBB detector is loaded from
# PRETRAINED_DETECTOR and remains frozen; only the relation stack is trained.
cfg["MODEL"]["FREEZE_BACKBONE"] = True
cfg["MODEL"]["FREEZE_NECK"] = True
cfg["MODEL"]["FREEZE_RPN_HEAD"] = True
cfg["MODEL"]["FREEZE_ROI_HEAD"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["BATCH_SIZE_PER_IMAGE"] = 512
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_FEAT_UPDATE_STEP"] = 3
# Legacy root alias retained for direct comparison with the SGG-Toolkit
# command line; the project predictor consumes RPCM_FEAT_UPDATE_STEP above.
cfg["feat_update_step"] = 3
# "pred" is standard sgcls. Set it to "gt" only for legacy filter ablations;
# the relation predictor still consumes detector logits in both cases.
cfg["MODEL"]["ROI_RELATION_HEAD"]["SGCLS_FILTER_LABEL_SOURCE"] = os.environ.get(
    "SGCLS_FILTER_LABEL_SOURCE", "pred"
).lower()
_filter_method = os.environ.get("FILTER_METHOD", "PPG").upper()
if _filter_method not in {"PPG", "PPN", "RSGP"}:
    raise ValueError("FILTER_METHOD must be PPG, PPN, or RSGP; unfiltered pair graphs are disabled.")
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"] = _filter_method
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_ENABLED"] = _filter_method == "PPG"
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_ENABLED"] = _filter_method == "PPN"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_ENABLED"] = _filter_method == "RSGP"
# The PPG score is evaluated before its top-k reduction.  The original
# all-pair protocol remains intact, while this chunk cap keeps very large
# STAR GT-box images bounded on GPU.
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_CHUNK_SIZE"] = int(
    os.environ.get("PPG_CHUNK_SIZE", "100000")
)

# Direct aliases of the original script's optimization contract.
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE", "8"))
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = int(os.environ.get("VAL_BATCH_SIZE", "1"))
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE", "1"))
cfg["SOLVER"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TRAIN_BATCH_SIZE"]
cfg["TEST"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TEST_BATCH_SIZE"]
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "1e-3"))
cfg["SOLVER"]["MAX_ITER"] = int(os.environ.get("MAX_ITER", "10000"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "100000"))
cfg["SOLVER"]["STEPS"] = [6000, 8500]
cfg["SOLVER"]["ITERATION_COMPAT"] = True
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2000"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "1000"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get("OUTPUT_DIR", "outputs/star_sgcls_obb_train")
