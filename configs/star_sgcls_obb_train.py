"""Main configuration for STAR OBB Scene Graph Classification (SGCls).

Task protocol: GT OBBs are provided, while both object labels and predicates
are predicted by the model.

The default reproduces the SGG-Toolkit/STAR comparison protocol:
  * the detector produces object logits on GT boxes;
  * the relation head's out_obj module refines object labels;
  * the pair filter reads GT labels by default
    (SGCLS_FILTER_LABEL_SOURCE=gt).

The last item affects only candidate pairs from Semantic Filter/PPG/PPN/RSGP;
it does not write GT labels into the final object predictions. Set
SGCLS_FILTER_LABEL_SOURCE=pred for a strict fully predicted-label ablation.

Typical launch command: bash scripts/run_star_sgcls_experiment.sh
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
# 1. SGCls task: GT boxes + predicted object labels + predicted predicates
# ---------------------------------------------------------------------------
cfg = copy.deepcopy(_base_cfg)
cfg["MODEL"]["TASK"] = "sgcls"
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_BOX"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_OBJECT_LABEL"] = False
# Unlike PredCls, SGCls learns the RPCM object-refinement classifier.
cfg["MODEL"]["ROI_RELATION_HEAD"]["OBJECT_REFINE_LOSS_WEIGHT"] = 1.0
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
# Match SGG-Toolkit/STAR's legacy sgcls protocol by default: proposals are GT
# boxes, detector logits/pred_labels are attached, but proposal.labels remains
# the GT label consumed by Semantic Filter / PPG / RSGP.  Use
# SGCLS_FILTER_LABEL_SOURCE=pred only for strict predicted-label ablations.
cfg["MODEL"]["ROI_RELATION_HEAD"]["SGCLS_FILTER_LABEL_SOURCE"] = os.environ.get(
    "SGCLS_FILTER_LABEL_SOURCE", "gt"
).lower()
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
# The PPG score is evaluated before its top-k reduction.  The original
# all-pair protocol remains intact, while this chunk cap keeps very large
# STAR GT-box images bounded on GPU.
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_CHUNK_SIZE"] = int(
    os.environ.get("PPG_CHUNK_SIZE", "100000")
)

# ---------------------------------------------------------------------------
# 2. Source-compatible iteration-based optimization protocol
# MAX_ITER is the effective hard stop. MAX_EPOCHS is deliberately large and
# serves only as a safety ceiling. With ITERATION_COMPAT=True, STEPS,
# VAL_START_PERIOD, and VAL_PERIOD are optimizer-step counts rather than epoch
# counts. Logs still print epochs to show dataset traversal progress.
# ---------------------------------------------------------------------------
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = int(os.environ.get("VAL_BATCH_SIZE", "1"))
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE", "1"))
cfg["SOLVER"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TRAIN_BATCH_SIZE"]
cfg["TEST"]["IMS_PER_BATCH"] = cfg["DATALOADER"]["TEST_BATCH_SIZE"]
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "1e-3"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = os.environ.get("LR_SCALE_BY_BATCH", "1") != "0"
cfg["SOLVER"]["MAX_ITER"] = int(os.environ.get("MAX_ITER", "15000"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "100000"))
# Current sgcls continuation has already passed step 12000 and was prematurely
# decayed again at 16000.  Keep the first decay active when resuming from the
# current checkpoint, but push the later decays back.  For a fresh run, override
# with STEPS=18000,25500 if you want the original 60%/85% schedule.
_steps = os.environ.get("STEPS", "8000,13000")
cfg["SOLVER"]["STEPS"] = [int(item) for item in _steps.replace(";", ",").split(",") if item.strip()]
cfg["SOLVER"]["ITERATION_COMPAT"] = True
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "100"))
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "10000"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "0"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get("OUTPUT_DIR", "outputs/star_sgcls_obb_train")
