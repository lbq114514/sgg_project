from copy import deepcopy

from configs.star_predcls_obb_train import cfg as base_cfg


cfg = deepcopy(base_cfg)
rel = cfg["MODEL"]["ROI_RELATION_HEAD"]
rel["PREDICTOR"] = "TYPED_HYPER_RPCM"
rel["TEST_FILTER_METHOD"] = "PPN"
rel["PPN_ENABLED"] = True
rel["PPN_MODEL_PATH"] = "pretrained/PPN_OBB.pth"
rel["GRAPH_TOPK"] = 10000
rel["GRAPH_PARTITION_SIZE"] = 2000
rel["GRAPH_MAX_DEGREE"] = 48
rel["GRAPH_RELAXED_MAX_DEGREE"] = 64
rel["GRAPH_GT_INJECT_FULL_EPOCHS"] = 5
rel["GRAPH_GT_INJECT_END_EPOCH"] = 10
rel["GRAPH_CACHE_PATH"] = ""
rel["TYPED_GRAPH_HIDDEN_DIM"] = 512
rel["TYPED_SPARSE_LAYERS"] = 0
rel["TYPED_ANCHOR_TOPK"] = 4
rel["TYPED_FAMILY_LOSS_WEIGHT"] = 0.5
rel["TYPED_ANCHOR_LOSS_WEIGHT"] = 0.2
rel["TYPED_LOGIC_LOSS_WEIGHT"] = 0.1
rel["TYPED_HIERARCHY_LOGIT_WEIGHT"] = 1.0
rel["TYPED_PROTO_LOGIT_WEIGHT"] = 0.0
rel["TYPED_HYPERGRAPH_ENABLED"] = False
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 4
cfg["SOLVER"]["GRADIENT_ACCUMULATION_STEPS"] = 4
cfg["SOLVER"]["OUTPUT_DIR"] = "outputs/star_predcls_obb_typed_stage1"
cfg["TEST"]["RECALL_AT"] = [1500, 2000]
