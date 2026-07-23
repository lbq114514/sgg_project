from copy import deepcopy


def get_default_cfg():
    cfg = {
        "MODEL": {
            "BOX_MODE": "hbb",  # hbb | obb
            "OBB_ANGLE_UNIT": "degree",  # degree | radian
            "TASK": "sgdet",  # predcls | sgcls | sgdet
            "NUM_CLASSES": 49,
            "NUM_PREDICATES": 59,
            "RPN_ONLY": False,
            "RETINANET_ON": False,
            "RELATION_ON": True,
            "ATTRIBUTE_ON": False,
            "MASK_ON": False,
            "KEYPOINT_ON": False,
            "FREEZE_BACKBONE": False,
            "FREEZE_NECK": False,
            "FREEZE_RPN_HEAD": False,
            "FREEZE_ROI_HEAD": False,
            # Original SGG-Toolkit OBB detector compatibility.  The fields
            # mirror its mmrotate train/test config while keeping the public
            # SceneGraphDetector API unchanged.
            "SGDET_COMPAT": {
                "ENABLED": False,
                "FREEZE_DETECTOR": True,
                "USE_D2": True,
                "D2_SCALE": 0.5,
                "RPN_ANCHOR_SIZES": [32, 64, 128, 256, 512],
                "RPN_ASPECT_RATIOS": [0.5, 1.0, 2.0],
                # mmdet AnchorGenerator default used by the source STAR
                # detector.  A 0.5 offset shifts every FPN anchor by half a
                # stride and is incompatible with its pretrained RPN deltas.
                "RPN_ANCHOR_OFFSET": 0.0,
                "RPN_NMS_PRE": 2000,
                "RPN_MAX_PER_IMG": 2000,
                "RPN_NMS_THRESH": 0.8,
                "RCNN_SCORE_THRESH": 0.05,
                "RCNN_NMS_THRESH": 0.1,
                "RCNN_MAX_PER_IMG": 2000,
                "PATCH_MERGE_NMS_THRESH": 0.4,
                "TRAIN_LABEL_SOURCE": "matched_gt",  # matched_gt | pred
                "EVAL_LABEL_SOURCE": "matched_gt",  # matched_gt | pred
                "PATCH_DEBUG": False,
                # The original STAR Sgdets branch calls the frozen detector's
                # ``simple_test`` path in both train and test.  GT participates
                # in proposal matching/relation supervision, but is not
                # appended to the final detector proposal set.
                "ADD_GTBOX_TO_PROPOSAL_IN_TRAIN": False,
                # Frozen-detector acceleration for sgdet only.  The cache is
                # generated offline and stores post-NMS proposals, not GT or
                # relation targets. REQUIRE_HIT prevents accidental fallback
                # to the multi-minute raw detector path.
                "DETECTION_CACHE": {
                    "ENABLED": False,
                    "DIR": "outputs/star_sgdet_detection_cache_v5",
                    "REQUIRE_HIT": True,
                    "HASH": "",
                },
            },
            # Object-class channel order of MODEL.PRETRAINED_DETECTOR.  The
            # project itself always stores background at channel zero; keep
            # this default to avoid changing custom/internal checkpoints.
            "PRETRAINED_DETECTOR_CLASS_ORDER": "background_first",
            "BACKBONE": {
                "NAME": "resnet_backbone",
                "OUT_CHANNELS": 256,
                "RESNET_NAME": "resnet50",
                "PRETRAINED": False,
            },
            "NECK": {
                "NAME": "fpn_neck",
                "IN_CHANNELS": [256, 512, 1024, 2048],
                "OUT_CHANNELS": 256,
            },
            "PROPOSAL_GENERATOR": {
                "NAME": "dummy_proposal_generator",
                "NUM_PROPOSALS": 64,
            },
            "ROI_EXTRACTOR": {
                "NAME": "simple_roi_extractor",
                "OUT_CHANNELS": 256,
                "POOL_SIZE": 7,
                "FEATMAP_NAMES": ["p2", "p3", "p4", "p5"],
                "FEATMAP_STRIDES": [4, 8, 16, 32],
                "FINEST_SCALE": 56,
            },
            "ROI_BOX_HEAD": {
                "FC_DIM": 1024,
                "NUM_CLASSES": 49,
                "BOX_LOSS_WEIGHT": 1.0,
                "CLS_LOSS_WEIGHT": 1.0,
            },
            "ROI_HEADS": {
                "FG_IOU_THRESHOLD": 0.5,
                "BG_IOU_THRESHOLD": 0.3,
                "BATCH_SIZE_PER_IMAGE": 256,
                "POSITIVE_FRACTION": 0.5,
                "DETECTIONS_PER_IMG": 80,
            },
            "ROI_RELATION_HEAD": {
                "USE_GT_BOX": False,
                "USE_GT_OBJECT_LABEL": False,
                "NUM_CLASSES": 59,
                "BATCH_SIZE_PER_IMAGE": 512,
                "MAX_TEST_PAIRS_PER_IMAGE": 10000,
                "TEST_PAIR_SAMPLER": "ALL",
                "TEST_SUBGRAPH_TOPK": 32,
                "TEST_SUBGRAPH_GLOBAL_TOPK": 0,
                "TEST_SUBGRAPH_COMPLETION_TOPK": 0,
                "POSITIVE_FRACTION": 0.25,
                "PREDICT_USE_VISION": True,
                "PREDICT_USE_BIAS": False,
                "BIAS_LAMBDA": 0.0,
                "REQUIRE_BOX_OVERLAP": False,
                # sgcls only: labels supplied to Semantic Filter and the pair
                # proposal filter at evaluation.  "pred" is task-faithful;
                # "gt" reproduces the original SGG-Toolkit filtering path.
                "SGCLS_FILTER_LABEL_SOURCE": "pred",  # pred | gt
                "PREDICTOR": "UNIMPLEMENTED",
                "CONTEXT_HIDDEN_DIM": 512,
                "CONTEXT_POOLING_DIM": 256,
                "EMBED_DIM": 200,
                "WORD_EMBEDDING_FEATURES": True,
                "EDGE_FEATURES_REPRESENTATION": "fusion",
                "USE_PAIR_SCALE_ENCODER": True,
                "UNION_MULTI_SCALE_POOLING": True,
                "RPCM_MLP_DIM": 1024,
                "RPCM_FEAT_UPDATE_STEP": 1,
                "RPCM_DROPOUT": 0.2,
                "RPCM_MARGIN": 0.2,
                "RPCM_USE_PROTOTYPE": True,
                "RPCM_PROTO_MOMENTUM": 0.9,
                "RPCM_PROTO_INIT": "semantic",
                "RPCM_PROTO_EMBED_DIM": 0,
                "RPCM_PROTO_GLOVE_PATH": "",
                # ``rpcm`` reproduces the historical RPCM predicate polarity
                # construction and raw-scale object GloVe embeddings.
                "RPCM_GLOVE_INIT_MODE": "current",
                "RPCM_OBJ_PROTO_INIT": "semantic",
                "RPCM_PAIR_LABEL_PRIOR": True,
                "RPCM_PAIR_PRIOR_DIM": 128,
                "PROTO_SEMANTIC_PUSH_ENABLED": True,
                "PROTO_ANTONYM_PAIRS": [],
                "PROTO_COMPETITOR_PAIRS": [],
                "PROTO_LAMBDA_PULL": 0.2,
                "PROTO_LAMBDA_SEP": 0.002,
                "PROTO_LAMBDA_ANT_SEP": 0.05,
                "PROTO_LAMBDA_COMP_SEP": 0.005,
                "PROTO_ANT_SEP_MARGIN": -0.10,
                "PROTO_COMP_SEP_MARGIN": 0.10,
                "PROTO_SEP_TYPE": "etf",
                "PROTO_TEXT_INIT_MODIFIER_AWARE": True,
                "SEMANTIC_GLOVE_PATH": "glove/glove.6B.200d.txt",
                "NUM_SUBGRAPH_QUERIES": 8,
                "SUBGRAPH_ATTENTION_HEADS": 4,
                "SUBGRAPH_DROPOUT": 0.1,
                "HIER_USE_PROTOTYPE": True,
                "HIER_PROTO_MOMENTUM": 0.9,
                "HIER_PAIRNESS_SCORE_WEIGHT": 1.0,
                "HIER_MARGIN": 0.2,
                "PREDICATE_LOSS_TYPE": "ce",  # ce | class_balanced | logit_adjusted
                "PREDICATE_CLASS_BALANCED_BETA": 0.999,
                "PREDICATE_BG_LOSS_WEIGHT": 1.0,
                "PREDICATE_LOGIT_ADJUST_TAU": 1.0,
                "PREDICATE_AUX_LOGIT_ADJUST_WEIGHT": 0.0,
                "PREDICATE_AUX_LOGIT_ADJUST_TAU": 0.5,
                "PREDICATE_COUNTS": [],
                "NUM_SAMPLE_PER_GT_REL": 4,
                "TAIL_AWARE_POS_SAMPLING": False,
                "TAIL_AWARE_POS_ALPHA": 0.5,
                "TAIL_AWARE_POS_MIN_COUNT": 1.0,
                # A dense all-pairs relation graph is not a supported runtime
                # mode for STAR: large OBB images can make it quadratic in the
                # number of entities before RPCM starts. Every task config
                # must choose PPG, PPN or RSGP explicitly; PPG is the safe
                # legacy default.
                "TEST_FILTER_METHOD": "PPG",
                "SEMA_F_ENABLED": False,
                "SEMA_F_PATH": "pretrained/SF_list_support.json",
                "PPG_ENABLED": True,
                "PPG_PAIR_THRESHOLD": 10000,
                "PPG_TOPK": 10000,
                "TEST_FILTER_TOPK": 10000,
                "PPG_CHUNK_SIZE": 100000,
                "PPG_ENCODING_DIM": 25,
                "PPG_HIDDEN_DIM1": 50,
                "PPG_HIDDEN_DIM2": 50,
                "PPG_MODEL_PATH_OBB": "pretrained/STAR_OBB.pth",
                "PPG_MODEL_PATH_HBB": "pretrained/STAR_HBB.pth",
                "PPN_ENABLED": False,
                "PPN_MODEL_PATH": "outputs/star_pair_proposal_network/model_best.pth",
                "PPN_PAIR_THRESHOLD": 10000,
                "PPN_TOPK": 10000,
                "PPN_CHUNK_SIZE": 200000,
                "RSGP_ENABLED": False,
                "RSGP_MODE": "HYBRID",  # RS_ONLY | PPN_GRAPH | HYBRID
                "RSGP_THRESHOLD": 10000,
                "RSGP_TOPK": 10000,
                "RSGP_CHUNK_SIZE": 200000,
                "RSGP_PPG_PROTECTED_TOPK": 7000,
                "RSGP_PPN_POOL_TOPK": 12000,
                "RSGP_RS_POOL_TOPK": 12000,
                "RSGP_MAX_OUT_DEGREE": 96,
                "RSGP_MAX_IN_DEGREE": 96,
                "RSGP_RELAXED_MAX_DEGREE": 128,
                "RSGP_LABEL_PAIR_QUOTA": 800,
                "RSGP_RELAXED_LABEL_PAIR_QUOTA": 1200,
                "RSGP_W_PPG": 1.0,
                "RSGP_W_PPN": 0.35,
                "RSGP_W_GEOM": 0.35,
                "RSGP_W_ANCHOR": 0.25,
                "RSGP_W_TOPO": 0.20,
                "RSGP_W_TAIL": 0.15,
                "RSGP_W_DEGREE": 0.15,
                # Component switches used for inference-only RSGP ablations.
                # Defaults preserve the full RSGP behavior.
                "RSGP_USE_PPN_COMPLETION": True,
                "RSGP_USE_GEOMETRY": True,
                "RSGP_USE_ANCHOR": True,
                "RSGP_USE_TOPOLOGY": True,
                "RSGP_USE_TAIL_PRIOR": True,
                "RSGP_USE_DEGREE_SCORE": True,
                "RSGP_ENFORCE_DEGREE_CAP": True,
                "RSGP_ENFORCE_LABEL_QUOTA": True,
                "RSGP_ANCHOR_CLASSES": "apron,truck_parking,car_parking,dock,runway,taxiway,breakwater,goods_yard",
                "RSGP_VEHICLE_CLASSES": "airplane,aircraft,vehicle,car,truck,ship,boat,bus,van",
                "RSGP_NETWORK_CLASSES": "tower,lattice_tower,substation,genset,transmission_line,power_line,line,pole",
                "RSGP_TAIL_PREDICATES": [7, 14, 20, 24, 25, 28, 31, 33, 36, 38, 39, 41, 53, 56, 58],
                "RPCM_LEGACY_FILTER_FLOW": False,
                "RPCM_LEGACY_UNION_BOX": False,
                # Relation-to-relation topology used by RPCM_LEGACY and
                # RPCM_ORIGINAL_LEGACY.  ``unified`` reproduces STAR's RCA
                # convention: two relation nodes are adjacent whenever they
                # share either endpoint (including subject/object cross-role
                # matches).  ``dual_view`` keeps shared-subject and
                # shared-object graphs separate.  Both modes reuse the same
                # GCN parameters, so changing this flag does not change the
                # checkpoint structure.
                "RPCM_RELATION_GRAPH_MODE": "dual_view",  # sgg_toolkit | unified | dual_view
                "RPCM_REL_SUBJ_VIEW_ENABLED": True,
                "RPCM_REL_OBJ_VIEW_ENABLED": True,
                # Exact training behavior of RPCM/weights/6850_4135.pth:
                # K=1 prototypes, static initialization EMA, historic antonym
                # loss, and layer-averaged dual-view object/relation states.
                "RPCM_LEGACY_6850_EXACT": False,
                "RPCM_LEGACY_ANT_LOSS_WEIGHT": 0.1,
                "RPCM_LEGACY_ANT_MARGIN": -0.2,
                "CAUSAL": {
                    "SPATIAL_FOR_VISION": True,
                },
            },
            "RELATION_HEAD": {
                "NAME": "latent_subgraph_relation_head",
                "NODE_DIM": 256,
                "EDGE_DIM": 256,
                "HIDDEN_DIM": 256,
                "NUM_PREDICATES": 59,
                "NUM_SUBGRAPHS": 8,
                "NUM_INTRA_LAYERS": 2,
                "NUM_INTER_LAYERS": 2,
                "NUM_ATTENTION_HEADS": 4,
                "MAX_REL_PAIRS": 256,
                "USE_GEOMETRY": True,
            },
        },
        "DATASETS": {
            "TRAIN": {
                "NAME": "generic_sgg_json",
                "ANN_FILE": "",
                "IMAGE_ROOT": "",
                "FILTER_EMPTY_RELATIONS": False,
                "TILE_ENABLED": False,
                "TILE_SIZE": [1024, 1024],
                "TILE_STRIDE": [768, 768],
                "TILE_CONTEXT": [0, 0],
                "TILE_MIN_OBJECTS": 1,
            },
            "VAL": {
                "NAME": "generic_sgg_json",
                "ANN_FILE": "",
                "IMAGE_ROOT": "",
                "FILTER_EMPTY_RELATIONS": False,
                "TILE_ENABLED": False,
                "TILE_SIZE": [1024, 1024],
                "TILE_STRIDE": [768, 768],
                "TILE_CONTEXT": [0, 0],
                "TILE_MIN_OBJECTS": 1,
            },
            "TEST": {
                "NAME": "generic_sgg_json",
                "ANN_FILE": "",
                "IMAGE_ROOT": "",
                "FILTER_EMPTY_RELATIONS": False,
                "TILE_ENABLED": False,
                "TILE_SIZE": [1024, 1024],
                "TILE_STRIDE": [768, 768],
                "TILE_CONTEXT": [0, 0],
                "TILE_MIN_OBJECTS": 1,
            },
        },
        "DATALOADER": {
            "BATCH_SIZE": 2,
            "TRAIN_BATCH_SIZE": 8,
            "VAL_BATCH_SIZE": 4,
            "TEST_BATCH_SIZE": 4,
            "NUM_WORKERS": 6,
            "SIZE_DIVISIBLE": 0,
        },
        "SOLVER": {
            "GRADIENT_ACCUMULATION_STEPS": 1,
            "OPTIMIZER": "SGD",
            "BASE_LR": 1e-4,
            "WEIGHT_DECAY": 1e-4,
            "WEIGHT_DECAY_BIAS": 1e-4,
            "BIAS_LR_FACTOR": 1.0,
            "BETAS": (0.9, 0.999),
            "EPS": 1e-8,
            "MOMENTUM": 0.9,
            "WARMUP_ITERS": 0,
            "WARMUP_EPOCHS": 0,
            "WARMUP_FACTOR": 0.001,
            "WARMUP_METHOD": "linear",
            "MAX_EPOCHS": 12,
            "STEPS": [],
            "GAMMA": 0.1,
            "CHECKPOINT_PERIOD": 1,
            "VAL_PERIOD": 1,
            "VAL_START_PERIOD": 0,
            "VAL_SPLIT": "val",
            "PRINT_GRAD_FREQ": 0,
            "PRINT_TRAIN_STEP_FREQ": 0,
            "PRINT_TRAIN_BATCH_FREQ": 0,
            "GRAD_NORM_CLIP": 5.0,
            "SCHEDULE": {
                "TYPE": "WarmupMultiStepLR",
                "UNIT": "iter",
                "MAX_ITERS": 0,
                "MIN_LR_RATIO": 0.0,
                "EXP_GAMMA": 0.999,
                "PATIENCE": 2,
                "THRESHOLD": 0.001,
                "COOLDOWN": 0,
                "FACTOR": 0.1,
                "MAX_DECAY_STEP": 3,
            },
            "OUTPUT_DIR": "outputs/default",
        },
        "RUNTIME": {
            "DISABLE_CUDNN": True,
            "CUDNN_BENCHMARK": False,
            "CUDNN_DETERMINISTIC": True,
        },
        "TEST": {
            "RECALL_AT": [20, 50, 100],
            "IOU_THRESHOLD": 0.5,
            "RELATION": {
                "LATER_NMS_PREDICTION_THRES": 0.3,
            },
            "EVAL_DEBUG": {
                "ENABLED": False,
                "TOP_PREDICATES": 10,
                "TOP_IMAGES": 10,
                "PRINT_HARDEST_IMAGES": False,
                "PRINT_CANDIDATE_COVERAGE": False,
                "PRINT_PAIR_CONFUSION": False,
                "PRINT_VEHICLE_AUX": False,
            },
            "GRAPH_DEBUG": {
                "ENABLED": False,
            },
            "TILE_MERGE_IOU_THRESHOLD": 0.6,
            "PATCH_AUTO_ENABLED": True,
            "PATCH_AUTO_MIN_SIZE": 1024,
            "PATCH_MAX_PYRAMID_LAYERS": 8,
            "PATCH_BATCH_SIZE": 2,
            "PATCH_BATCH_SIZE_LARGE": 4,
            "PATCH_GAPS": [200],
            "PATCH_SIZE": [1024, 1024],
            "PATCH_SCORE_THRESHOLDS": [0.3, 0.2, 0.1, 0.001, 0.00001],
        },
        "TYPE": "CV",
        "mbs": 512,
        "feat_update_step": 4,
        "EXP_nums": 30,
    }
    return deepcopy(cfg)
