import os

from configs.star_predcls_obb_train import cfg


# Legacy RPCM path intended to match the original RPCM repository launch:
#
#   /home/ubuntu/research/ssd/RPCM/Scripts/LOBB_RPCM_predcls_train.sh
#
# The goal is not to use the newer Typed/PPN graph code, but to run the closest
# native RPCM + PPG baseline inside this project for controlled comparison.

rel = cfg["MODEL"]["ROI_RELATION_HEAD"]

cfg["MODEL"]["TASK"] = "predcls"
cfg["MODEL"]["BOX_MODE"] = "obb"
cfg["MODEL"]["OBB_ANGLE_UNIT"] = "radian"
cfg["MODEL"]["USE_COMBINED_ROI_HEADS"] = True
cfg["MODEL"]["FREEZE_BACKBONE"] = True
cfg["MODEL"]["FREEZE_NECK"] = True
cfg["MODEL"]["FREEZE_RPN_HEAD"] = True
cfg["MODEL"]["FREEZE_ROI_HEAD"] = True
cfg["MODEL"]["PRETRAINED_DETECTOR"] = "pretrained/OBB_swin_L_OBD.pth"
cfg["MODEL"]["STORE_DETECTOR_D2"] = True

rel["PREDICTOR"] = "RPCM_ORIGINAL_LEGACY"
rel["USE_GT_BOX"] = True
rel["USE_GT_OBJECT_LABEL"] = True
rel["TEST_FILTER_METHOD"] = "PPG"
rel["PPG_ENABLED"] = True
rel["PPG_MODEL_PATH_OBB"] = "pretrained/STAR_OBB.pth"
rel["PPG_PAIR_THRESHOLD"] = 10000
rel["PPG_TOPK"] = 10000
rel["PPG_CHUNK_SIZE"] = 1000000
rel["PPN_ENABLED"] = False
rel["RPCM_LEGACY_FILTER_FLOW"] = True
rel["RPCM_LEGACY_UNION_BOX"] = True
rel["PREDICT_USE_BIAS"] = os.environ.get("PREDICT_USE_BIAS", "0") == "1"
rel["BIAS_LAMBDA"] = 0.2
rel["RPCM_LEGACY_BIAS_LAMBDA_TRAIN"] = 0.4
rel["RPCM_LEGACY_BIAS_LAMBDA_TEST"] = 0.5

rel["SEMA_F_ENABLED"] = True
rel["SEMA_F_PATH"] = os.environ.get(
    "SEMA_F_PATH",
    "/home/ubuntu/research/ssd/RPCM/maskrcnn_benchmark/modeling/roi_heads/relation_head/SF_list.json",
)
rel["TEST_PAIR_SAMPLER"] = "ALL"
rel["MAX_TEST_PAIRS_PER_IMAGE"] = 0

# Original relation sampler settings from LOBB_RPCM_predcls_train.sh:
#   MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512
# The original base YAML has POSITIVE_FRACTION=0.25, but mbs=512 makes the
# predcls positive cap effectively 512 in the original sampler.
rel["BATCH_SIZE_PER_IMAGE"] = 512
rel["POSITIVE_FRACTION"] = 0.25
cfg["mbs"] = 512

# Native RPCM settings. The original implementation hard-codes mlp_dim=2048;
# feat_update_step is 4 in the saved original config.
rel["RPCM_USE_PROTOTYPE"] = True
rel["RPCM_MLP_DIM"] = 2048
rel["RPCM_FEAT_UPDATE_STEP"] = 4
rel["RPCM_DROPOUT"] = 0.2
rel["RPCM_PAIR_LABEL_PRIOR"] = False
rel["PREDICATE_LOSS_TYPE"] = "ce"
rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
rel["PROTO_SEMANTIC_PUSH_ENABLED"] = False
rel["PROTO_ANTONYM_PAIRS"] = []
rel["PROTO_COMPETITOR_PAIRS"] = []

# Original RPCM uses 200D object word embeddings in PairwiseFeatureExtractor,
# but initializes predicate prototypes from 300D GloVe.
rel["EMBED_DIM"] = 200
rel["SEMANTIC_GLOVE_PATH"] = "glove/glove.6B.200d.txt"
rel["PAIRWISE_EMBED_EXTRA_BACKGROUND"] = False
rel["RPCM_PROTO_EMBED_DIM"] = 300
rel["RPCM_PROTO_GLOVE_PATH"] = "glove/glove.6B.300d.txt"

# The original YAML sets CONTEXT_POOLING_DIM=4096. This increases memory
# substantially; use 4096 for closest matching. Override to 2048 at launch if
# GPU memory is insufficient.
rel["CONTEXT_POOLING_DIM"] = 4096
cfg["MODEL"]["RELATION_HEAD"]["NODE_DIM"] = 4096
cfg["MODEL"]["RELATION_HEAD"]["EDGE_DIM"] = 4096

# Match the original mmrotate detector ROI extractor:
#   roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2,
#                  clockwise=True)
#   finest_scale=16
#
# The project default was inherited from a generic FPN ROI extractor
# (sampling_ratio=0, clockwise=False, finest_scale=56).  That changes every
# object/union visual feature before RPCM sees it and is not checkpoint
# compatible with the original STAR OBB detector.
cfg["MODEL"]["ROI_EXTRACTOR"]["SAMPLING_RATIO"] = 2
cfg["MODEL"]["ROI_EXTRACTOR"]["CLOCKWISE"] = True
cfg["MODEL"]["ROI_EXTRACTOR"]["FINEST_SCALE"] = 16

# Match original launch:
#   SOLVER.IMS_PER_BATCH 16
#   SOLVER.MAX_ITER 17000
#   SOLVER.BASE_LR 1e-3
#   SOLVER.STEPS "(10000, 13000, 15000)"
# This trainer is epoch-based. The fixed STAR train split has 771 images, so
# batch size 16 gives ceil(771 / 16) = 49 optimizer steps per epoch. The original
# 17000-iter RPCM schedule therefore corresponds to ceil(17000 / 49) = 347
# epochs; use 350 by default to cover the full schedule. The LR is intentionally
# not multiplied by TRAIN_BATCH_SIZE for legacy matching.
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 16
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = int(os.environ.get("VAL_BATCH_SIZE", "1"))
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE", "1"))
cfg["DATALOADER"]["NUM_WORKERS"] = 8
cfg["DATALOADER"]["SIZE_DIVISIBLE"] = 32

cfg["SOLVER"]["OPTIMIZER"] = "SGD"
cfg["SOLVER"]["BASE_LR"] = 1e-3
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = False
cfg["SOLVER"]["MOMENTUM"] = 0.9
cfg["SOLVER"]["WEIGHT_DECAY"] = 0.0001
cfg["SOLVER"]["WEIGHT_DECAY_BIAS"] = 0.0
cfg["SOLVER"]["WARMUP_ITERS"] = 500
cfg["SOLVER"]["WARMUP_EPOCHS"] = 0
cfg["SOLVER"]["WARMUP_FACTOR"] = 0.1
cfg["SOLVER"]["WARMUP_METHOD"] = "linear"
cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupMultiStepLR"
cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "iter"
cfg["SOLVER"]["STEPS"] = [10000, 13000, 15000]
cfg["SOLVER"]["GAMMA"] = 0.1
cfg["SOLVER"]["GRAD_NORM_CLIP"] = 5.0
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "4"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "4"))
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "286"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "350"))
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR",
    "outputs/star_predcls_obb_rpcm_legacy_ppg_aligned",
)

cfg["TEST"]["RECALL_AT"] = [1500, 2000]
cfg["TEST"]["EVAL_DEBUG"]["ENABLED"] = True

# Original mm_config's effective train pipeline uses RResize + directional flip
# + Normalize + Pad; PolyRandomRotate is not in the final data.train.pipeline.
cfg["DATASETS"]["TRAIN"]["ROTATE_RATIO"] = 0.0
for _split in ("TRAIN", "VAL", "TEST"):
    cfg["DATASETS"][_split]["KEEP_RATIO"] = True
    # Original RPCM's STAR OBB loader hard-codes
    #   poly[id] = poly[id] / BOX_SCALE * 6000
    # before mmrotate RResize.  The generic loader uses max(width, height),
    # which changes every GT OBB and all downstream PPG/ROI/union features.
    cfg["DATASETS"][_split]["BOX_COORD_SCALE"] = 6000.0
cfg["feat_update_step"] = 4
