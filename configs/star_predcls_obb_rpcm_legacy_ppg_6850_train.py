import os
from copy import deepcopy

from configs.star_predcls_obb_rpcm_legacy_ppg_train import cfg as _base_cfg


cfg = deepcopy(_base_cfg)


# Compatibility profile for:
#
#   /home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth
#
# Evidence from the checkpoint/log:
#   - saved at RPCM/nohup.out iter 17600
#   - test R@1500=0.6850, mR@1500=0.4135
#   - full training was 20000 iterations
#   - optimizer state is SGD/momentum, current lr=0.0016, initial_lr=0.016
#   - predictor has 88 keys:
#       * single 2D predicate prototypes: rel_proto.base_prototypes [59, 300]
#       * rel_proto.proto_ema [59, 2048]
#       * no rel_proto.proto_vis / proto_vis_inited
#       * no bias_module.* parameters

rel = cfg["MODEL"]["ROI_RELATION_HEAD"]

# Older 6850 checkpoint uses the pre-visual-prototype RPCM head.
rel["RPCM_LEGACY_NUM_PROTO"] = 1
rel["RPCM_LEGACY_USE_VIS_PROTO"] = False
rel["RPCM_LEGACY_PROTO_2D_COMPAT"] = True
rel["RPCM_LEGACY_REGISTER_BIAS_MODULE"] = False

# Keep the rest of the legacy predictor aligned with the original run.
rel["PREDICT_USE_BIAS"] = False
rel["BIAS_LAMBDA"] = 0.2
rel["RPCM_FEAT_UPDATE_STEP"] = 4
rel["RPCM_MLP_DIM"] = 2048
rel["EMBED_DIM"] = 200
rel["PAIRWISE_EMBED_EXTRA_BACKGROUND"] = False
rel["RPCM_PROTO_EMBED_DIM"] = 300
rel["RPCM_PROTO_GLOVE_PATH"] = "glove/glove.6B.300d.txt"

# The old log shows warmup to lr=0.016, then drops to 0.0016 around 12k and
# 0.00016 after 18k.  The current trainer has a different batch-scaling rule, so
# use the effective LR directly and disable another scale-by-batch pass.
cfg["SOLVER"]["OPTIMIZER"] = "SGD"
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = False
cfg["SOLVER"]["MOMENTUM"] = 0.9
cfg["SOLVER"]["WEIGHT_DECAY"] = 0.0001
cfg["SOLVER"]["WEIGHT_DECAY_BIAS"] = 0.0
cfg["SOLVER"]["WARMUP_ITERS"] = 500
cfg["SOLVER"]["WARMUP_FACTOR"] = 0.1
cfg["SOLVER"]["SCHEDULE"]["TYPE"] = "WarmupMultiStepLR"
cfg["SOLVER"]["SCHEDULE"]["UNIT"] = "iter"
cfg["SOLVER"]["STEPS"] = [12000, 18000]
cfg["SOLVER"]["GAMMA"] = 0.1

# 771 fixed train images / batch 16 ~= 49 iter per epoch; 20000 iter ~= 409
# epochs.  Use 410 to cover the old run length.
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "410"))
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "245"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "4"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "test")
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR",
    "outputs/star_predcls_obb_rpcm_legacy_ppg_6850",
)
