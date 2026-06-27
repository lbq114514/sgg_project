import torch

from sgg.config.defaults import get_default_cfg
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.structures.boxes import BoxList


def apply_runtime_cfg(cfg):
    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def random_targets(batch_size=2, n_obj=8, mode="hbb"):
    targets = []
    for _ in range(batch_size):
        if mode == "hbb":
            centers = torch.rand(n_obj, 2) * 384.0 + 64.0
            sizes = torch.rand(n_obj, 2) * 96.0 + 32.0
            half = sizes * 0.5
            boxes = torch.cat([centers - half, centers + half], dim=1)
        else:
            centers = torch.rand(n_obj, 2) * 384.0 + 64.0
            sizes = torch.rand(n_obj, 2) * 96.0 + 32.0
            angles = (torch.rand(n_obj, 1) - 0.5) * 90.0
            boxes = torch.cat([centers, sizes, angles], dim=1)
        labels = torch.randint(1, 10, (n_obj,))
        pair_labels = torch.randint(0, 6, (n_obj, n_obj))
        boxlist_mode = "xyxy" if mode == "hbb" else "xywha"
        target = BoxList(boxes, image_size=(512, 512), mode=boxlist_mode)
        target.add_field("labels", labels)
        target.add_field("pair_labels", pair_labels)
        triplets = []
        for s in range(n_obj):
            for o in range(n_obj):
                if s != o and pair_labels[s, o] > 0:
                    triplets.append([s, o, int(pair_labels[s, o].item())])
        if triplets:
            target.add_field("relation_triplets", torch.tensor(triplets, dtype=torch.long))
        else:
            target.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long))
        targets.append(target)
    return targets


def main():
    cfg = get_default_cfg()
    cfg["MODEL"]["BOX_MODE"] = "obb"  # switch between "hbb" and "obb"
    cfg["MODEL"]["TASK"] = "sgdet"
    cfg["MODEL"]["NUM_CLASSES"] = 10
    cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"] = 10
    cfg["MODEL"]["NUM_PREDICATES"] = 6
    cfg["MODEL"]["RELATION_HEAD"]["NUM_PREDICATES"] = 6
    cfg["MODEL"]["BACKBONE"] = {
        "NAME": "swin",
        "EMBED_DIMS": 32,
        "PATCH_SIZE": 4,
        "WINDOW_SIZE": 4,
        "MLP_RATIO": 4,
        "DEPTHS": (2, 2, 2, 2),
        "NUM_HEADS": (2, 4, 8, 16),
        "OUT_INDICES": (0, 1, 2, 3),
        "PATCH_NORM": True,
        "DROP_RATE": 0.0,
        "ATTN_DROP_RATE": 0.0,
        "DROP_PATH_RATE": 0.0,
        "WITH_CP": False,
    }
    cfg["MODEL"]["NECK"] = {
        "NAME": "fpn_neck",
        "IN_CHANNELS": [32, 64, 128, 256],
        "OUT_CHANNELS": 64,
        "NUM_OUTS": 4,
        "UPSAMPLE_MODE": "nearest",
    }
    cfg["MODEL"]["ROI_EXTRACTOR"]["OUT_CHANNELS"] = 64
    cfg["MODEL"]["ROI_EXTRACTOR"]["FEATURE_KEY"] = "p2"
    cfg["MODEL"]["ROI_EXTRACTOR"]["POOL_SIZE"] = 7
    cfg["MODEL"]["ROI_EXTRACTOR"]["FEATURE_STRIDE"] = 4
    cfg["MODEL"]["ROI_EXTRACTOR"]["ANGLE_VERSION"] = "oc"
    cfg["MODEL"]["DETECTION_FEATURE_KEY"] = "p2"
    cfg["MODEL"]["RELATION_HEAD"]["NODE_DIM"] = 64
    cfg["MODEL"]["RELATION_HEAD"]["EDGE_DIM"] = 64
    cfg["MODEL"]["RELATION_HEAD"]["HIDDEN_DIM"] = 64
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["NUM_PROPOSALS"] = 32
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["PRE_NMS_TOPK"] = 128
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["SIZES"] = ((32,),)
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["ASPECT_RATIOS"] = (0.5, 1.0, 2.0)
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["ANGLES"] = (-45.0, 0.0, 45.0, 90.0)
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["FG_IOU_THRESHOLD"] = 0.5
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["BG_IOU_THRESHOLD"] = 0.2
    cfg["MODEL"]["PROPOSAL_GENERATOR"]["OBB_FALLBACK_TO_HBB"] = False
    apply_runtime_cfg(cfg)
    model = SceneGraphDetector(cfg)

    images = torch.randn(2, 3, 512, 512)
    targets = random_targets(batch_size=2, n_obj=10, mode=cfg["MODEL"]["BOX_MODE"])
    model.train()
    out = model(images, targets)
    print("Train keys:", out.keys())
    print({k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else type(v)) for k, v in out.items()})

    model.eval()
    with torch.no_grad():
        preds = model(images)
    print("Eval batch size:", len(preds))
    print("Prediction fields:", preds[0].fields())


if __name__ == "__main__":
    main()
