"""Clean PredCls training config for HPRC (historical code name: tail_aux).

Fresh runs initialize only the frozen detector from
``pretrained/OBB_swin_L_OBD.pth``.  The RPCM relation stack, prototypes and
HPRC head are initialized from the corrected canonical configuration; no
RPCM relation checkpoint is implied by this file.

The ``RPCM_TAIL_AUX_*`` keys are retained for checkpoint and command-line
compatibility.  In the paper this module is called Hard-Predicate Residual
Calibration (HPRC).
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)

rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

# Detector-only initialization contract for fresh training.  SceneGraphDetector
# loads this checkpoint before Trainer starts; no relation-head tensors are
# loaded by the launcher.  PredCls keeps the complete detector frozen.
cfg["MODEL"]["PRETRAINED_DETECTOR"] = "pretrained/OBB_swin_L_OBD.pth"
cfg["MODEL"]["PRETRAINED_DETECTOR_CLASS_ORDER"] = "background_last"
cfg["MODEL"]["FREEZE_BACKBONE"] = True
cfg["MODEL"]["FREEZE_NECK"] = True
cfg["MODEL"]["FREEZE_RPN_HEAD"] = True
cfg["MODEL"]["FREEZE_ROI_HEAD"] = True

# Pin the corrected RPCM-6850 architecture explicitly.  These assignments are
# intentionally repeated here so a future change to another PredCls experiment
# cannot silently move the HPRC scratch baseline back to an earlier graph,
# prototype or GloVe implementation.
rel_cfg["PREDICTOR"] = "RPCM_ORIGINAL_LEGACY"
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "dual_view"
rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_GLOVE_INIT_MODE"] = "rpcm"
rel_cfg["RPCM_LEGACY_NUM_PROTO"] = 1
rel_cfg["RPCM_LEGACY_USE_VIS_PROTO"] = False
rel_cfg["RPCM_LEGACY_PROTO_2D_COMPAT"] = True
rel_cfg["RPCM_LEGACY_6850_EXACT"] = True
rel_cfg["RPCM_LEGACY_ANT_LOSS_WEIGHT"] = float(
    os.environ.get("RPCM_LEGACY_ANT_LOSS_WEIGHT", "0.1")
)
rel_cfg["RPCM_LEGACY_ANT_MARGIN"] = float(
    os.environ.get("RPCM_LEGACY_ANT_MARGIN", "-0.2")
)

# HPRC keeps the main CE objective and adds weak logit-adjust supervision plus
# a zero-initialized residual calibration head.
rel_cfg["PREDICATE_LOSS_TYPE"] = "ce"
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = float(
    os.environ.get("PREDICATE_AUX_LOGIT_ADJUST_WEIGHT", "0.1")
)
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_TAU"] = float(
    os.environ.get("PREDICATE_AUX_LOGIT_ADJUST_TAU", "0.5")
)
# Hard predicates selected from training frequency and stable validation
# confusion/recall patterns.  They are not all frequency-tail classes. IDs
# follow STAR predicate IDs with 0 reserved for background.
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = True
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = [
    7,   # randomly docked at
    14,  # randomly parked on
    20,  # not run along
    24,  # not parked alongside with
    25,  # running along the different taxiway with
    28,  # running along the same taxiway with
    31,  # within danger distance of
    33,  # incorrectly parked on
    36,  # not docked alongside with
    38,  # driving in the different lane with
    39,  # driving in the same lane with
    41,  # driving alongside with
    53,  # indirectly connected to
    56,  # indirectly transmit electricity to
    58,  # not working on
]
rel_cfg["RPCM_TAIL_AUX_HIDDEN_DIM"] = int(os.environ.get("RPCM_TAIL_AUX_HIDDEN_DIM", "512"))
rel_cfg["RPCM_TAIL_AUX_DROPOUT"] = float(os.environ.get("RPCM_TAIL_AUX_DROPOUT", "0.1"))
rel_cfg["RPCM_TAIL_AUX_LOSS_WEIGHT"] = float(os.environ.get("RPCM_TAIL_AUX_LOSS_WEIGHT", "0.2"))
rel_cfg["RPCM_TAIL_AUX_LOGIT_MAX_WEIGHT"] = float(
    os.environ.get("RPCM_TAIL_AUX_LOGIT_MAX_WEIGHT", "0.3")
)
rel_cfg["RPCM_TAIL_AUX_LOGIT_INIT"] = float(os.environ.get("RPCM_TAIL_AUX_LOGIT_INIT", "0.0"))

# PredCls returns fixed GT one-hot object logits.  Their cross entropy is a
# constant with no gradient, so exclude it from the clean PredCls loss/log.
# SGCls and SGDet explicitly restore this weight to 1.0 in their own configs.
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0

# Full detector-only scratch schedule.  STEPS are optimizer iterations while
# validation/checkpoint periods are epochs for PredCls.
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = False
cfg["SOLVER"]["WARMUP_ITERS"] = int(os.environ.get("WARMUP_ITERS", "500"))
cfg["SOLVER"]["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "10000,14000,16000").split(",")
    if step.strip()
]
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "120"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "4"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "300"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR",
    "outputs/star_predcls_obb_hprc_scratch",
)
