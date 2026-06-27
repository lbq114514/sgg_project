from copy import deepcopy

from configs.star_predcls_obb_train import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_ENABLED"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"] = "NONE"
cfg["SOLVER"]["OUTPUT_DIR"] = "outputs/star_predcls_obb_ablate_pairness_score"
