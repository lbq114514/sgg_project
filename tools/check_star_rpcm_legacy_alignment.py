from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.data.build import build_dataset
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector


def load_py_config(path: str) -> Dict:
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def all_ordered_pairs(num_obj: int, device: torch.device) -> torch.Tensor:
    if num_obj <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    ids = torch.arange(num_obj, device=device)
    subj = ids[:, None].expand(num_obj, num_obj).reshape(-1)
    obj = ids[None, :].expand(num_obj, num_obj).reshape(-1)
    mask = subj != obj
    return torch.stack((subj[mask], obj[mask]), dim=1)


def gt_pair_recall(candidate_pairs: torch.Tensor, relations: torch.Tensor) -> tuple[int, int, float]:
    if relations.numel() == 0:
        return 0, 0, 0.0
    gt_pairs = {(int(s), int(o)) for s, o, _ in relations.tolist() if int(s) != int(o)}
    cand = {(int(s), int(o)) for s, o in candidate_pairs.tolist()}
    hit = len(gt_pairs & cand)
    total = len(gt_pairs)
    return hit, total, hit / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check current project vs original RPCM legacy assumptions on STAR.")
    parser.add_argument("--config", default="configs/star_predcls_obb_rpcm_legacy_ppg_train.py")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_py_config(args.config)
    cfg["DATALOADER"]["NUM_WORKERS"] = 0
    dataset = build_dataset(cfg, args.split)
    meta = dataset.metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [meta.categories[i] for i in sorted(meta.categories)]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [meta.predicates[i] for i in sorted(meta.predicates)]

    model = SceneGraphDetector(cfg).to(args.device)
    model.eval()
    relation_head = model.roi_heads.relation

    state_keys = list(model.state_dict().keys())
    d2_keys = [k for k in state_keys if any(part.endswith("d2") for part in k.split("."))]
    report = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "predictor": cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICTOR"],
        "detector_d2": {
            "enabled": bool(cfg["MODEL"].get("STORE_DETECTOR_D2", False)),
            "state_keys": len(d2_keys),
            "has_backbone_d2": model.backbone_d2 is not None,
            "has_neck_d2": model.neck_d2 is not None,
            "has_rpn_head_d2": model.rpn_head_d2 is not None,
            "has_roi_head_d2": model.roi_head_d2 is not None,
        },
        "ppg": {
            "enabled": relation_head.ppg.enabled,
            "loaded": relation_head.ppg.loaded,
            "filter_method": relation_head.ppg.filter_method,
            "topk": relation_head.ppg.topk,
            "threshold": relation_head.ppg.threshold,
        },
        "samples": [],
    }

    for idx in range(min(int(args.num_images), len(dataset))):
        image, target, meta_item = dataset[idx]
        image = image.unsqueeze(0).to(args.device)
        proposal = target.to(args.device)
        labels = proposal.get_field("labels").long()
        all_pairs = all_ordered_pairs(len(proposal), proposal.bbox.device)
        sem_pairs = relation_head.sema_filter.filter_pairs(all_pairs, labels)
        ppg_pairs = relation_head.ppg.filter_pairs(proposal, sem_pairs)
        relations = (
            proposal.get_field("all_relation_triplets")
            if proposal.has_field("all_relation_triplets")
            else proposal.get_field("relation_triplets")
        )

        with torch.inference_mode():
            features = model._extract_features(image)
            roi_feats = model.bbox_roi_extractor(features, [proposal])
            bbox_features = model.bbox_head(roi_feats, flag=True)
            bbox_features_compat = model._bbox_forward(features, [proposal], flag=True)
            d2_feature_shapes = {}
            if model.backbone_d2 is not None:
                d2_features = model._extract_features_d2(image)
                d2_feature_shapes = {key: list(value.shape) for key, value in d2_features.items()}

        angle_stats = None
        if proposal.mode == "xywha" and proposal.bbox.numel() > 0:
            angles = proposal.bbox[:, 4].detach().float().cpu()
            angle_stats = {
                "min": float(angles.min().item()),
                "max": float(angles.max().item()),
                "abs_max": float(angles.abs().max().item()),
                "looks_like_degrees": bool(angles.abs().max().item() > 3.2),
            }

        all_hit, all_total, all_recall = gt_pair_recall(all_pairs.cpu(), relations.cpu())
        sem_hit, sem_total, sem_recall = gt_pair_recall(sem_pairs.cpu(), relations.cpu())
        ppg_hit, ppg_total, ppg_recall = gt_pair_recall(ppg_pairs.cpu(), relations.cpu())
        sample_report = {
            "dataset_index": idx,
            "image_id": int(meta_item.get("image_id", -1)),
            "num_objects": len(proposal),
            "num_relations": int(relations.size(0)),
            "angle_stats": angle_stats,
            "candidate_counts": {
                "all_non_self": int(all_pairs.size(0)),
                "semantic": int(sem_pairs.size(0)),
                "ppg": int(ppg_pairs.size(0)),
            },
            "gt_pair_recall": {
                "all_non_self": {"hit": all_hit, "total": all_total, "recall": all_recall},
                "semantic": {"hit": sem_hit, "total": sem_total, "recall": sem_recall},
                "ppg": {"hit": ppg_hit, "total": ppg_total, "recall": ppg_recall},
            },
            "feature_shapes": {
                "roi_feats": list(roi_feats.shape),
                "bbox_features": list(bbox_features.shape),
                "bbox_features_compat": list(bbox_features_compat.shape),
                "d2": d2_feature_shapes,
            },
            "bbox_feature_compat_max_abs_diff": float(
                (bbox_features - bbox_features_compat).abs().max().item()
            ) if bbox_features.numel() else 0.0,
        }
        report["samples"].append(sample_report)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
