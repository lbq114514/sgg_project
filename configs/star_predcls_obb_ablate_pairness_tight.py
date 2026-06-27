from copy import deepcopy

from configs.star_predcls_obb_train import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_TOPK"] = 4000
cfg["SOLVER"]["OUTPUT_DIR"] = "outputs/star_predcls_obb_ablate_pairness_tight"
