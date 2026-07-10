from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)

rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

# Keep the original RPCM legacy branch as the base model. This branch only adds
# low-risk tail/modifier auxiliary supervision and residual logits.
rel_cfg["PREDICATE_LOSS_TYPE"] = "ce"
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = float(
    os.environ.get("PREDICATE_AUX_LOGIT_ADJUST_WEIGHT", "0.1")
)
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_TAU"] = float(
    os.environ.get("PREDICATE_AUX_LOGIT_ADJUST_TAU", "0.5")
)

# Low-recall predicates with enough test support. IDs follow STAR predicate IDs
# with 0 reserved for background.
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

# Earlier validation makes the LR/tail behavior observable before the first
# decay. Keep these overridable for long/full runs.
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "80"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "200"))
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR",
    "outputs/star_predcls_obb_tail_aux",
)
