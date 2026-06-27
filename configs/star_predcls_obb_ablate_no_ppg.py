from copy import deepcopy

from configs.star_predcls_obb_train import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg["MODEL"]["ROI_RELATION_HEAD"]["PPG_ENABLED"] = False
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"] = "SCORE_TOPK"
cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_TOPK"] = 10000
cfg["SOLVER"]["OUTPUT_DIR"] = "outputs/star_predcls_obb_ablate_no_ppg"
