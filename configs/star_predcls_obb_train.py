"""Main configuration for STAR OBB Predicate Classification (PredCls).

Task protocol: GT OBBs and GT object labels are provided; the model predicts
only predicates between entity pairs. The complete detector/backbone remains
part of the checkpoint but is frozen in PredCls. It supplies RoI/union visual
features rather than generating boxes or object labels.

Common runtime overrides:
    FILTER_METHOD=PPG|PPN|RSGP
    OUTPUT_DIR=outputs/...
    BASE_LR=... MAX_EPOCHS=...
    VAL_START_PERIOD=... VAL_PERIOD=...

Important: STAR images may contain many entities. FILTER_METHOD must not be
empty because an all-pairs graph can cause OOM.
"""

import os

from sgg.config.defaults import get_default_cfg


DATA_ROOT = os.environ.get(
    "STAR_SGG_ROOT",
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG",
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_list_int(name: str, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# 1. Task protocol and detector freezing policy
# ---------------------------------------------------------------------------
cfg = get_default_cfg()
cfg["MODEL"]["TASK"] = "predcls"
cfg["MODEL"]["BOX_MODE"] = "obb"
cfg["MODEL"]["OBB_ANGLE_UNIT"] = "radian"
cfg["MODEL"]["USE_COMBINED_ROI_HEADS"] = True
cfg["MODEL"]["FREEZE_BACKBONE"] = True
cfg["MODEL"]["FREEZE_NECK"] = True
cfg["MODEL"]["FREEZE_RPN_HEAD"] = True
cfg["MODEL"]["FREEZE_ROI_HEAD"] = True
cfg["MODEL"]["STORE_DETECTOR_D2"] = True

# ---------------------------------------------------------------------------
# 2. Relation predictor and prototype settings
# RPCM_ORIGINAL_LEGACY is the main route and preserves the original RPCM
# feature, GCN, and prototype logic.
# ---------------------------------------------------------------------------
cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICTOR"] = "RPCM_ORIGINAL_LEGACY"
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_BOX"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["USE_GT_OBJECT_LABEL"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_USE_PROTOTYPE"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_SEMANTIC_PUSH_ENABLED"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_ANTONYM_PAIRS"] = []
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_COMPETITOR_PAIRS"] = []
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_LAMBDA_PULL"] = 0.2
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_LAMBDA_SEP"] = 0.01
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_LAMBDA_ANT_SEP"] = 0.1
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_LAMBDA_COMP_SEP"] = 0.01
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_ANT_SEP_MARGIN"] = -0.20
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_COMP_SEP_MARGIN"] = 0.20
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_SEP_TYPE"] = "etf"
cfg["MODEL"]["ROI_RELATION_HEAD"]["PROTO_TEXT_INIT_MODIFIER_AWARE"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["SEMANTIC_GLOVE_PATH"] = "glove/glove.6B.200d.txt"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_PAIR_LABEL_PRIOR"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_PAIR_PRIOR_DIM"] = 128
cfg["MODEL"]["ROI_RELATION_HEAD"]["HIER_PAIRNESS_SCORE_WEIGHT"] = 0.0
cfg["MODEL"]["ROI_RELATION_HEAD"]["CONTEXT_HIDDEN_DIM"] = 512
cfg["MODEL"]["ROI_RELATION_HEAD"]["CONTEXT_POOLING_DIM"] = 4096
cfg["MODEL"]["ROI_RELATION_HEAD"]["EMBED_DIM"] = 200
cfg["MODEL"]["ROI_RELATION_HEAD"]["WORD_EMBEDDING_FEATURES"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["CAUSAL"]["SPATIAL_FOR_VISION"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["UNION_ROI_ALIGN_CHUNK_SIZE"] = 1024
cfg["MODEL"]["ROI_RELATION_HEAD"]["BATCH_SIZE_PER_IMAGE"] = 512
cfg["MODEL"]["ROI_RELATION_HEAD"]["POSITIVE_FRACTION"] = 0.25
cfg["MODEL"]["ROI_RELATION_HEAD"]["MAX_TEST_PAIRS_PER_IMAGE"] = 0
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_PAIR_SAMPLER"] = "ALL"
# ---------------------------------------------------------------------------
# 3. Pair filter
# PPG: original paper baseline; PPN: independent label+OBB network;
# RSGP: hybrid graph using PPG, PPN, and remote-sensing priors.
# This stage selects candidate pairs only. RPCM still predicts predicates.
# ---------------------------------------------------------------------------
_filter_method = os.environ.get(
    "FILTER_METHOD",
    os.environ.get("TEST_FILTER_METHOD", "PPG"),
).upper()
if _filter_method not in {"PPG", "PPN", "RSGP"}:
    raise ValueError(
        "FILTER_METHOD must be PPG, PPN, or RSGP; "
        "unfiltered all-pairs relation graphs are disabled for STAR."
    )
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"] = _filter_method
# This setting is only consumed by TASK="sgcls".  Keep the task-faithful
# predicted-label behavior as the default for future sgcls configs.
cfg["MODEL"]["ROI_RELATION_HEAD"]["SGCLS_FILTER_LABEL_SOURCE"] = os.environ.get(
    "SGCLS_FILTER_LABEL_SOURCE", "pred"
).lower()
cfg["MODEL"]["ROI_RELATION_HEAD"]["SEMA_F_ENABLED"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["SEMA_F_PATH"] = os.environ.get(
    "SEMA_F_PATH",
    "pretrained/SF_list_support.json",
)
# Legacy mirror fields only. Runtime filter selection is controlled by
# TEST_FILTER_METHOD/FILTER_METHOD.
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_ENABLED"] = _filter_method == "PPG"
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_ENABLED"] = _filter_method == "PPN"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_ENABLED"] = _filter_method == "RSGP"
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_PAIR_THRESHOLD"] = _env_int("PPG_PAIR_THRESHOLD", 10000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_TOPK"] = _env_int("PPG_TOPK", 10000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_CHUNK_SIZE"] = _env_int("PPG_CHUNK_SIZE", 1000000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_BASE_PAIR_KEEP_TOPK"] = 0
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_SUBGRAPH_COMPLETION_TOPK"] = 0
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_PAIRNESS_TOPK"] = 0
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_PAIRNESS_COMPLETION_TOPK"] = 0
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_ENCODING_DIM"] = 25
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_HIDDEN_DIM1"] = 50
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_HIDDEN_DIM2"] = 50
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_MODEL_PATH_OBB"] = os.environ.get(
    "PPG_MODEL_PATH_OBB",
    "pretrained/STAR_OBB.pth",
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_MODEL_PATH_HBB"] = os.environ.get(
    "PPG_MODEL_PATH_HBB",
    "pretrained/STAR_HBB.pth",
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_MODEL_PATH"] = os.environ.get(
    "PPN_MODEL_PATH",
    "pretrained/PPN_OBB.pth",
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_PAIR_THRESHOLD"] = _env_int("PPN_PAIR_THRESHOLD", 10000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_TOPK"] = _env_int("PPN_TOPK", 10000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPN_CHUNK_SIZE"] = _env_int("PPN_CHUNK_SIZE", 200000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_MODE"] = os.environ.get("RSGP_MODE", "HYBRID").upper()
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_THRESHOLD"] = _env_int("RSGP_THRESHOLD", 10000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_TOPK"] = _env_int("RSGP_TOPK", 10000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_CHUNK_SIZE"] = _env_int("RSGP_CHUNK_SIZE", 200000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_PPG_PROTECTED_TOPK"] = _env_int(
    "RSGP_PPG_PROTECTED_TOPK",
    7000,
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_PPN_POOL_TOPK"] = _env_int("RSGP_PPN_POOL_TOPK", 12000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_RS_POOL_TOPK"] = _env_int("RSGP_RS_POOL_TOPK", 12000)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_MAX_OUT_DEGREE"] = _env_int("RSGP_MAX_OUT_DEGREE", 96)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_MAX_IN_DEGREE"] = _env_int("RSGP_MAX_IN_DEGREE", 96)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_RELAXED_MAX_DEGREE"] = _env_int("RSGP_RELAXED_MAX_DEGREE", 128)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_LABEL_PAIR_QUOTA"] = _env_int("RSGP_LABEL_PAIR_QUOTA", 800)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_RELAXED_LABEL_PAIR_QUOTA"] = _env_int(
    "RSGP_RELAXED_LABEL_PAIR_QUOTA",
    1200,
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_PPG"] = _env_float("RSGP_W_PPG", 1.0)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_PPN"] = _env_float("RSGP_W_PPN", 0.35)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_GEOM"] = _env_float("RSGP_W_GEOM", 0.35)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_ANCHOR"] = _env_float("RSGP_W_ANCHOR", 0.25)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_TOPO"] = _env_float("RSGP_W_TOPO", 0.20)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_TAIL"] = _env_float("RSGP_W_TAIL", 0.15)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_W_DEGREE"] = _env_float("RSGP_W_DEGREE", 0.15)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_USE_PPN_COMPLETION"] = _env_bool(
    "RSGP_USE_PPN_COMPLETION", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_USE_GEOMETRY"] = _env_bool(
    "RSGP_USE_GEOMETRY", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_USE_ANCHOR"] = _env_bool(
    "RSGP_USE_ANCHOR", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_USE_TOPOLOGY"] = _env_bool(
    "RSGP_USE_TOPOLOGY", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_USE_TAIL_PRIOR"] = _env_bool(
    "RSGP_USE_TAIL_PRIOR", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_USE_DEGREE_SCORE"] = _env_bool(
    "RSGP_USE_DEGREE_SCORE", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_ENFORCE_DEGREE_CAP"] = _env_bool(
    "RSGP_ENFORCE_DEGREE_CAP", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_ENFORCE_LABEL_QUOTA"] = _env_bool(
    "RSGP_ENFORCE_LABEL_QUOTA", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_ANCHOR_CLASSES"] = os.environ.get(
    "RSGP_ANCHOR_CLASSES",
    "apron,truck_parking,car_parking,dock,runway,taxiway,breakwater,goods_yard",
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_VEHICLE_CLASSES"] = os.environ.get(
    "RSGP_VEHICLE_CLASSES",
    "airplane,aircraft,vehicle,car,truck,ship,boat,bus,van",
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_NETWORK_CLASSES"] = os.environ.get(
    "RSGP_NETWORK_CLASSES",
    "tower,lattice_tower,substation,genset,transmission_line,power_line,line,pole",
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RSGP_TAIL_PREDICATES"] = _env_list_int(
    "RSGP_TAIL_PREDICATES",
    [7, 14, 20, 24, 25, 28, 31, 33, 36, 38, 39, 41, 53, 56, 58],
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICATE_LOSS_TYPE"] = "ce"
cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_MLP_DIM"] = 2048
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_FEAT_UPDATE_STEP"] = 4
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_DROPOUT"] = 0.2
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_PAIR_LABEL_PRIOR"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["PAIRWISE_EMBED_EXTRA_BACKGROUND"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_PROTO_EMBED_DIM"] = 300
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_PROTO_GLOVE_PATH"] = "glove/glove.6B.300d.txt"
# Reproduce the GloVe construction used to train RPCM 6850_4135: raw 200-D
# object embeddings in PairwiseFeatureExtractor and polarity-aware normalized
# 300-D predicate prototypes. This affects scratch initialization only;
# checkpoint-loaded embeddings are still restored from the checkpoint.
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_GLOVE_INIT_MODE"] = "rpcm"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_FILTER_FLOW"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_UNION_BOX"] = True
# ``unified`` is the STAR/SGG-ToolKit relation adjacency. ``dual_view`` is the
# role-aware variant that propagates shared-subject and shared-object messages
# separately. The switch changes no parameter names or tensor shapes.
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_RELATION_GRAPH_MODE"] = os.environ.get(
    "RPCM_RELATION_GRAPH_MODE", "dual_view"
).lower()
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_REL_SUBJ_VIEW_ENABLED"] = _env_bool(
    "RPCM_REL_SUBJ_VIEW_ENABLED", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_REL_OBJ_VIEW_ENABLED"] = _env_bool(
    "RPCM_REL_OBJ_VIEW_ENABLED", True
)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICT_USE_BIAS"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["BIAS_LAMBDA"] = 0.2
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_BIAS_LAMBDA_TRAIN"] = 0.4
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_BIAS_LAMBDA_TEST"] = 0.5
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_NUM_PROTO"] = 1
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_USE_VIS_PROTO"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_PROTO_2D_COMPAT"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_REGISTER_BIAS_MODULE"] = False
# Reproduce the exact relation-head version that generated 6850_4135.pth.
# This keeps its single prototype, static proto_ema anchor, historic antonym
# objective, and dual-view layer averaging.  It adds no checkpoint tensors, so
# existing 6850/tail-aux/SGCls/SGDet checkpoints remain structurally loadable.
# PredCls inference is unchanged; SGCls/SGDet must still be one-shot evaluated
# because their object-refinement path consumes the restored entity average.
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_6850_EXACT"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_ANT_LOSS_WEIGHT"] = 0.1
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_LEGACY_ANT_MARGIN"] = -0.2

# ---------------------------------------------------------------------------
# 4. Frozen OBB detector and RoI feature-extractor architecture
# These dimensions must match pretrained/OBB_swin_L_OBD.pth and the RPCM
# input dimensions.
# ---------------------------------------------------------------------------
cfg["MODEL"]["BACKBONE"] = {
    "NAME": "swin",
    "EMBED_DIMS": 192,
    "PATCH_SIZE": 4,
    "WINDOW_SIZE": 7,
    "MLP_RATIO": 4,
    "DEPTHS": (2, 2, 18, 2),
    "NUM_HEADS": (6, 12, 24, 48),
    "OUT_INDICES": (0, 1, 2, 3),
    "PATCH_NORM": True,
    "DROP_RATE": 0.0,
    "ATTN_DROP_RATE": 0.0,
    "DROP_PATH_RATE": 0.2,
    "WITH_CP": True,
}
cfg["MODEL"]["NECK"] = {
    "NAME": "fpn_neck",
    "IN_CHANNELS": [192, 384, 768, 1536],
    "OUT_CHANNELS": 256,
    "NUM_OUTS": 5,
    "UPSAMPLE_MODE": "nearest",
}
cfg["MODEL"]["RPN_HEAD"] = {
    "NAME": "oriented_rpn_head",
    "IN_CHANNELS": 256,
    "FEAT_CHANNELS": 256,
    "NUM_ANCHORS": 3,
    "CLS_OUT_CHANNELS": 1,
    "VERSION": "le90",
    "USE_SIGMOID_CLS": True,
}
cfg["MODEL"]["PRETRAINED_DETECTOR"] = "pretrained/OBB_swin_L_OBD.pth"
# The original mmrotate STAR OBB detector stores its background classifier
# output in the last (48th) row.  SceneGraphDetector normalizes it to the
# project's background-first convention while loading.
cfg["MODEL"]["PRETRAINED_DETECTOR_CLASS_ORDER"] = "background_last"
cfg["MODEL"]["ROI_EXTRACTOR"]["NAME"] = "rotated_roi_extractor"
cfg["MODEL"]["ROI_EXTRACTOR"]["OUT_CHANNELS"] = 512
cfg["MODEL"]["ROI_EXTRACTOR"]["FEATURE_KEY"] = "p2"
cfg["MODEL"]["ROI_EXTRACTOR"]["POOL_SIZE"] = 7
cfg["MODEL"]["ROI_EXTRACTOR"]["FEATURE_STRIDE"] = 4
cfg["MODEL"]["ROI_EXTRACTOR"]["ANGLE_VERSION"] = "le90"
cfg["MODEL"]["ROI_EXTRACTOR"]["SAMPLING_RATIO"] = 2
cfg["MODEL"]["ROI_EXTRACTOR"]["CLOCKWISE"] = True
cfg["MODEL"]["ROI_EXTRACTOR"]["FINEST_SCALE"] = 16
cfg["MODEL"]["DETECTION_FEATURE_KEY"] = "p2"
cfg["MODEL"]["RELATION_HEAD"]["NODE_DIM"] = 4096
cfg["MODEL"]["RELATION_HEAD"]["EDGE_DIM"] = 4096
cfg["MODEL"]["RELATION_HEAD"]["HIDDEN_DIM"] = 512
cfg["MODEL"]["ROI_HEADS"]["FG_IOU_THRESHOLD"] = 0.5
cfg["MODEL"]["ROI_HEADS"]["BG_IOU_THRESHOLD"] = 0.3
cfg["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"] = 256
cfg["MODEL"]["ROI_HEADS"]["POSITIVE_FRACTION"] = 0.5
cfg["MODEL"]["ROI_HEADS"]["DETECTIONS_PER_IMG"] = 80
cfg["MODEL"]["ROI_BOX_HEAD"]["MLP_HEAD_DIM"] = 4096

# ---------------------------------------------------------------------------
# 5. Data loading and optimizer
# LR_SCALE_BY_BATCH=False means BASE_LR is the actual target learning rate;
# it is not automatically scaled with batch size.
# ---------------------------------------------------------------------------
cfg["DATALOADER"]["BATCH_SIZE"] = 8
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 16
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = int(os.environ.get("VAL_BATCH_SIZE", "2"))
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE", "2"))
cfg["DATALOADER"]["NUM_WORKERS"] = 8
cfg["DATALOADER"]["SIZE_DIVISIBLE"] = 32

cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = False
cfg["SOLVER"]["OPTIMIZER"] = "SGD"
cfg["SOLVER"]["WEIGHT_DECAY"] = 0.0001
cfg["SOLVER"]["WEIGHT_DECAY_BIAS"] = 0.0
cfg["SOLVER"]["BIAS_LR_FACTOR"] = 1.0
cfg["SOLVER"]["MOMENTUM"] = 0.9
cfg["SOLVER"]["WARMUP_ITERS"] = 500
cfg["SOLVER"]["WARMUP_EPOCHS"] = 0
cfg["SOLVER"]["WARMUP_FACTOR"] = 0.1
cfg["SOLVER"]["WARMUP_METHOD"] = "linear"
# PredCls is trained by epoch. MultiStep uses UNIT="iter", so STEPS contains
# optimizer-step indices, while VAL_START_PERIOD/VAL_PERIOD are epoch counts.
# Do not mix these two counters.
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "300"))
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get("OUTPUT_DIR", "outputs/star_predcls_obb_train")
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "4"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "4"))
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "245"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["PRINT_GRAD_FREQ"] = 0
cfg["SOLVER"]["GRAD_NORM_CLIP"] = 5.0
cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupMultiStepLR"
cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "iter"
cfg["SOLVER"]["STEPS"] = [12000, 16000, 18500]
cfg["SOLVER"]["GAMMA"] = 0.1
cfg["SOLVER"]["SCHEDULE"]["MIN_LR_RATIO"] = 0.0
cfg["SOLVER"]["SCHEDULE"]["EXP_GAMMA"] = 0.9999

# Optional learning-rate schedules:
# 1) MultiStep:
# cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupMultiStepLR"
# cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "epoch"
# cfg["SOLVER"]["STEPS"] = [260, 360]
# cfg["SOLVER"]["GAMMA"] = 0.1
#
# 2) Cosine:
# cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupCosineLR"
# cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "epoch"
# cfg["SOLVER"]["SCHEDULE"]["MIN_LR_RATIO"] = 0.01
#
# 3) Linear Decay:
# cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupLinearDecayLR"
# cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "epoch"
# cfg["SOLVER"]["SCHEDULE"]["MIN_LR_RATIO"] = 0.05
#
# 4) Exponential:
# cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupExponentialLR"
# cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "iter"
# cfg["SOLVER"]["SCHEDULE"]["EXP_GAMMA"] = 0.9999
#
# 5) Plateau:
# cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupReduceLROnPlateau"
# cfg["SOLVER"]["SCHEDULE"]["FACTOR"] = 0.5
# cfg["SOLVER"]["SCHEDULE"]["PATIENCE"] = 2
# cfg["SOLVER"]["SCHEDULE"]["THRESHOLD"] = 1e-4
# cfg["SOLVER"]["SCHEDULE"]["COOLDOWN"] = 0
#
# 6) Constant:
# cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "none"

# ---------------------------------------------------------------------------
# 6. Evaluation and large-image patch settings
# The STAR paper protocol reports R/mR/HMR@1000/1500/2000.
# ---------------------------------------------------------------------------
cfg["TEST"]["RECALL_AT"] = [1000, 1500, 2000]
cfg["TEST"]["EVAL_DEBUG"] = {
    "ENABLED": True,
    "TOP_PREDICATES": 10,
    "TOP_IMAGES": 10,
    "PRINT_HARDEST_IMAGES": _env_bool("EVAL_DEBUG_PRINT_HARDEST_IMAGES", False),
    "PRINT_CANDIDATE_COVERAGE": _env_bool("EVAL_DEBUG_PRINT_CANDIDATE_COVERAGE", False),
    "PRINT_PAIR_CONFUSION": _env_bool("EVAL_DEBUG_PRINT_PAIR_CONFUSION", False),
    "PRINT_VEHICLE_AUX": _env_bool("EVAL_DEBUG_PRINT_VEHICLE_AUX", False),
}
cfg["TEST"]["PATCH_AUTO_ENABLED"] = True
cfg["TEST"]["PATCH_AUTO_MIN_SIZE"] = 1024
cfg["TEST"]["PATCH_MAX_PYRAMID_LAYERS"] = 16
cfg["TEST"]["PATCH_BATCH_SIZE"] = 2
cfg["TEST"]["PATCH_BATCH_SIZE_LARGE"] = 4
cfg["TEST"]["PATCH_GAPS"] = [200]
cfg["TEST"]["PATCH_SIZE"] = [1024, 1024]
cfg["TEST"]["PATCH_SCORE_THRESHOLDS"] = [0.3, 0.2, 0.1, 0.001, 0.00001]
cfg["mbs"] = 512
cfg["feat_update_step"] = 4

# ---------------------------------------------------------------------------
# 7. STAR dataset
# The fixed split matches the source project. Relation deduplication keeps the
# single-predicate-per-pair GC Recall protocol. BOX_COORD_SCALE is the scene
# scale used for STAR OBB coordinate normalization.
# ---------------------------------------------------------------------------
for split in ("TRAIN", "VAL", "TEST"):
    cfg["DATASETS"][split]["NAME"] = "star"
    cfg["DATASETS"][split]["IMAGE_ROOT"] = f"{DATA_ROOT}/STAR_img"
    cfg["DATASETS"][split]["ROIDB_FILE"] = f"{DATA_ROOT}/STAR-SGG-with-attri.h5"
    cfg["DATASETS"][split]["DICT_FILE"] = f"{DATA_ROOT}/STAR-SGG-dicts-with-attri.json"
    cfg["DATASETS"][split]["IMAGE_FILE"] = f"{DATA_ROOT}/STAR_image_data_v1.json"
    cfg["DATASETS"][split]["FILTER_EMPTY_RELATIONS"] = True
    cfg["DATASETS"][split]["FILTER_DUPLICATE_RELATIONS"] = True
    cfg["DATASETS"][split]["FILTER_NON_OVERLAP"] = False
    cfg["DATASETS"][split]["SPLIT_MODE"] = "fixed"
    cfg["DATASETS"][split]["IMAGE_EXT"] = ".png"
    cfg["DATASETS"][split]["IMAGE_SIZE"] = [1024, 1024]
    cfg["DATASETS"][split]["PIXEL_MEAN"] = [123.675, 116.28, 103.53]
    cfg["DATASETS"][split]["PIXEL_STD"] = [58.395, 57.12, 57.375]
    cfg["DATASETS"][split]["TO_RGB"] = True
    cfg["DATASETS"][split]["AUGMENT"] = split == "TRAIN"
    cfg["DATASETS"][split]["KEEP_RATIO"] = True
    cfg["DATASETS"][split]["BOX_COORD_SCALE"] = 6000.0

cfg["DATASETS"]["TRAIN"]["ROTATE_RATIO"] = 0.0
