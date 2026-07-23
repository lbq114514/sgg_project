"""PredCls ablation A: original SGG-ToolKit heterogeneous GNN from scratch.

This branch keeps the same detector, RPCM parameters, prototype head, CE loss,
and PPG evaluation filter as the main experiment.  Only the relation-to-
relation adjacency is changed: relations sharing any endpoint are merged into
one graph, matching the topology convention used by the STAR baseline.

Only the frozen OBB detector is initialized from
``pretrained/OBB_swin_L_OBD.pth``. The relation predictor is not initialized
from an RPCM checkpoint.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

# Full SGG-ToolKit heterogeneous GNN, not merely the unified E x E adjacency:
# six collect modes, shared collect/update modules, complete per-image entity
# graph, unified relation graph, and last-step features.
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "sgg_toolkit"
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []

# Detector-only initialization requires the full relation-head training
# schedule. These defaults match the existing tail-aux main experiment so the
# cumulative ablations use the same optimization budget.
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["WARMUP_ITERS"] = int(os.environ.get("WARMUP_ITERS", "500"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "200"))
cfg["SOLVER"]["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "6000,8500,10000").split(",")
    if step.strip()
]
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "80"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "4"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/star_predcls_ablation_sgg_toolkit_gnn_scratch"
)
