from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector


def load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def apply_runtime_cfg(cfg):
    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def _pair_set(pair_idx: torch.Tensor) -> set[Tuple[int, int]]:
    if pair_idx.numel() == 0:
        return set()
    return {(int(subj), int(obj)) for subj, obj in pair_idx.tolist()}


def _proposal_label_field(proposal) -> torch.Tensor:
    if proposal.has_field("labels"):
        return proposal.get_field("labels").long()
    if proposal.has_field("pred_labels"):
        return proposal.get_field("pred_labels").long()
    return torch.zeros((len(proposal),), dtype=torch.long, device=proposal.bbox.device)


def _prepare_stage_pairs(model, relation_head, features, images, targets):
    proposals = model._prepare_eval_proposals(features, images, targets)
    device = next(iter(features.values())).device if isinstance(features, dict) else features.device
    base_pair_idxs = relation_head.samp_processor.prepare_test_pairs(device, proposals)
    sema_pair_idxs: List[torch.Tensor] = []
    final_pair_idxs: List[torch.Tensor] = []

    for proposal, base_pair_idx in zip(proposals, base_pair_idxs):
        sema_pair_idx = base_pair_idx
        labels = _proposal_label_field(proposal)
        if proposal.has_field("labels"):
            sema_pair_idx = relation_head.sema_filter.filter_pairs(base_pair_idx, labels)
        if relation_head.ppg.enabled and relation_head.ppg.filter_method == "PPG":
            final_pair_idx = relation_head.ppg.filter_pairs(proposal, sema_pair_idx)
        else:
            final_pair_idx = sema_pair_idx
        sema_pair_idxs.append(sema_pair_idx)
        final_pair_idxs.append(final_pair_idx)

    return proposals, base_pair_idxs, sema_pair_idxs, final_pair_idxs


def _stage_hits(gt_rels: torch.Tensor, pair_set: set[Tuple[int, int]]) -> Tuple[int, Dict[int, int]]:
    total_hits = 0
    per_pred_hits: Dict[int, int] = defaultdict(int)
    for subj, obj, pred in gt_rels.tolist():
        if (int(subj), int(obj)) in pair_set:
            total_hits += 1
            per_pred_hits[int(pred)] += 1
    return total_hits, per_pred_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/star_predcls_obb_train.py")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="test_gt_pair_coverage_ppg.json")
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--top-images", type=int, default=50)
    args = parser.parse_args()

    cfg = get_default_cfg()
    cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)

    datasets = build_datasets(cfg, splits=("test",))
    test_meta = datasets["test"].metadata
    predicate_names = [test_meta.predicates[i] for i in sorted(test_meta.predicates.keys())]
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        test_meta.categories[i] for i in sorted(test_meta.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = predicate_names

    model = SceneGraphDetector(cfg)
    dataloaders = build_dataloaders(
        cfg,
        splits=("test",),
        datasets=datasets,
        shuffle_map={"test": False},
    )
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    model.eval()
    relation_head = model.roi_heads.relation

    stage_names = ("base", "sema", "ppg")
    total_gt = 0
    stage_hit_totals = {name: 0 for name in stage_names}
    predicate_total: Dict[int, int] = defaultdict(int)
    predicate_hits = {name: defaultdict(int) for name in stage_names}
    image_rows = []

    with torch.no_grad():
        for batch_idx, (images, targets, metas) in enumerate(dataloaders["test"]):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = images.to(trainer.device)
            targets = trainer._move_targets(targets)
            features = model._extract_features(images)
            proposals, base_pair_idxs, sema_pair_idxs, final_pair_idxs = _prepare_stage_pairs(
                model, relation_head, features, images, targets
            )

            for proposal, target, meta, base_pair_idx, sema_pair_idx, final_pair_idx in zip(
                proposals, targets, metas, base_pair_idxs, sema_pair_idxs, final_pair_idxs
            ):
                gt_rels = (
                    target.get_field("relation_triplets").long()
                    if target.has_field("relation_triplets")
                    else torch.zeros((0, 3), dtype=torch.long, device=target.bbox.device)
                )
                if gt_rels.numel() == 0:
                    continue

                gt_rel_count = int(gt_rels.size(0))
                total_gt += gt_rel_count
                for pred in gt_rels[:, 2].tolist():
                    predicate_total[int(pred)] += 1

                stage_pair_sets = {
                    "base": _pair_set(base_pair_idx),
                    "sema": _pair_set(sema_pair_idx),
                    "ppg": _pair_set(final_pair_idx),
                }

                image_row = {
                    "image_id": int(meta.get("image_id", -1)),
                    "gt_rel_count": gt_rel_count,
                    "base_pair_count": int(base_pair_idx.size(0)),
                    "sema_pair_count": int(sema_pair_idx.size(0)),
                    "ppg_pair_count": int(final_pair_idx.size(0)),
                }

                for stage_name in stage_names:
                    hits, per_pred = _stage_hits(gt_rels, stage_pair_sets[stage_name])
                    stage_hit_totals[stage_name] += hits
                    image_row[f"{stage_name}_gt_hits"] = int(hits)
                    image_row[f"{stage_name}_gt_coverage"] = float(hits / gt_rel_count) if gt_rel_count > 0 else 0.0
                    for pred_id, count in per_pred.items():
                        predicate_hits[stage_name][pred_id] += int(count)
                image_rows.append(image_row)

    predicate_rows = []
    for pred_id in sorted(predicate_total.keys()):
        total = int(predicate_total[pred_id])
        row = {
            "predicate_id": int(pred_id),
            "predicate": str(predicate_names[pred_id]) if 0 <= pred_id < len(predicate_names) else str(pred_id),
            "count": total,
        }
        for stage_name in stage_names:
            hits = int(predicate_hits[stage_name].get(pred_id, 0))
            row[f"{stage_name}_hits"] = hits
            row[f"{stage_name}_coverage"] = float(hits / total) if total > 0 else 0.0
        predicate_rows.append(row)

    image_rows.sort(
        key=lambda item: (
            float(item["ppg_gt_coverage"]),
            float(item["sema_gt_coverage"]),
            float(item["base_gt_coverage"]),
            -int(item["gt_rel_count"]),
            int(item["image_id"]),
        )
    )

    output = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "task": str(cfg["MODEL"]["TASK"]),
        "filter_method": str(cfg["MODEL"]["ROI_RELATION_HEAD"].get("TEST_FILTER_METHOD", "NONE")),
        "sema_enabled": bool(cfg["MODEL"]["ROI_RELATION_HEAD"].get("SEMA_F_ENABLED", False)),
        "ppg_enabled": bool(cfg["MODEL"]["ROI_RELATION_HEAD"].get("PPG_ENABLED", False)),
        "summary": {
            "total_gt_relations": int(total_gt),
            "base_gt_hits": int(stage_hit_totals["base"]),
            "base_gt_coverage": float(stage_hit_totals["base"] / total_gt) if total_gt > 0 else 0.0,
            "sema_gt_hits": int(stage_hit_totals["sema"]),
            "sema_gt_coverage": float(stage_hit_totals["sema"] / total_gt) if total_gt > 0 else 0.0,
            "ppg_gt_hits": int(stage_hit_totals["ppg"]),
            "ppg_gt_coverage": float(stage_hit_totals["ppg"] / total_gt) if total_gt > 0 else 0.0,
        },
        "per_predicate": predicate_rows,
        "hardest_images": image_rows[: max(int(args.top_images), 0)],
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote GT pair coverage JSON to {output_path}")


if __name__ == "__main__":
    main()
