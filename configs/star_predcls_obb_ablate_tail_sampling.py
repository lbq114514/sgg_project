from copy import deepcopy

from configs.star_predcls_obb_train import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg["MODEL"]["ROI_RELATION_HEAD"]["TAIL_AWARE_POS_SAMPLING"] = True
cfg["MODEL"]["ROI_RELATION_HEAD"]["TAIL_AWARE_POS_ALPHA"] = 0.5
cfg["SOLVER"]["OUTPUT_DIR"] = "outputs/star_predcls_obb_ablate_tail_sampling"
