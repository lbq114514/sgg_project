from copy import deepcopy

from configs.star_predcls_obb_train import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_PROTO_INIT"] = "random"
cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_OBJ_PROTO_INIT"] = "random"
cfg["SOLVER"]["OUTPUT_DIR"] = "outputs/star_predcls_obb_ablate_random_proto"
