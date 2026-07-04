from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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


def _name_to_id(names: Sequence[str], target: str) -> int:
    for idx, name in enumerate(names):
        if name == target:
            return idx
    raise KeyError(f"Name not found: {target}")


def _pair_mask(pair_idx: torch.Tensor, subj: int, obj: int) -> torch.Tensor:
    if pair_idx.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=pair_idx.device)
    return (pair_idx[:, 0] == int(subj)) & (pair_idx[:, 1] == int(obj))


def _best_pair_score(pair_idx: torch.Tensor, logits: torch.Tensor, subj: int, obj: int):
    mask = _pair_mask(pair_idx, subj, obj)
    matched = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if matched.numel() == 0:
        return None
    picked = logits[matched]
    if picked.ndim == 1:
        return picked
    best = picked[:, 1:].max(dim=1).values.argmax()
    return picked[int(best)]


def _class_name(names: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(names):
        return str(names[idx])
    return str(idx)


def _prepare_relation_forward(model, relation_head, features, images, targets):
    proposals = model._prepare_eval_proposals(features, images, targets)
    device = next(iter(features.values())).device if isinstance(features, dict) else features.device
    rel_pair_idxs = relation_head.samp_processor.prepare_test_pairs(device, proposals)
    base_pair_idxs: List[torch.Tensor] = []
    final_pair_idxs: List[torch.Tensor] = []
    for proposal, pair_idx in zip(proposals, rel_pair_idxs):
        base_pair_idxs.append(pair_idx.clone())
        filtered = pair_idx
        if proposal.has_field("labels"):
            filtered = relation_head.sema_filter.filter_pairs(filtered, proposal.get_field("labels").long())
        if relation_head.ppg.enabled and relation_head.ppg.filter_method in {"PPG", "PPN"}:
            filtered = relation_head.ppg.filter_pairs(proposal, filtered)
        final_pair_idxs.append(filtered)

    if model is not None and hasattr(model, "bbox_roi_extractor") and hasattr(model, "bbox_head"):
        roi_feats = model.bbox_roi_extractor(features, list(proposals))
        roi_features = model.bbox_head(roi_feats)
    else:
        roi_features = relation_head.box_feature_extractor(features, proposals)
        roi_features = relation_head.local_box_feature_proj(roi_features)

    union_features = (
        relation_head.union_feature_extractor(features, proposals, final_pair_idxs, OBj=model)
        if relation_head.use_union_box
        else None
    )
    relation_logits, _ = relation_head.predictor(
        proposals,
        final_pair_idxs,
        None,
        None,
        roi_features,
        union_features,
        logger=None,
    )
    return proposals, base_pair_idxs, final_pair_idxs, relation_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/star_predcls_obb_train.py")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gt-predicate", type=str, default="parking in the different apron with")
    parser.add_argument(
        "--focus-predicates",
        nargs="+",
        default=[
            "parking in the same apron with",
            "parking in the different apron with",
            "parked alongside with",
            "not parked alongside with",
        ],
    )
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    cfg = get_default_cfg()
    cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)

    datasets = build_datasets(cfg, splits=("test",))
    test_meta = datasets["test"].metadata
    class_names = [test_meta.categories[i] for i in sorted(test_meta.categories.keys())]
    rel_names = [test_meta.predicates[i] for i in sorted(test_meta.predicates.keys())]
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = class_names
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = rel_names

    gt_pred_id = _name_to_id(rel_names, args.gt_predicate)
    focus_ids = [_name_to_id(rel_names, name) for name in args.focus_predicates]

    model = SceneGraphDetector(cfg)
    dataloaders = build_dataloaders(
        cfg,
        splits=("test",),
        datasets=datasets,
        shuffle_map={"test": False},
    )
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    trainer.load_checkpoint(args.checkpoint)
    model.eval()
    relation_head = model.roi_heads.relation

    total_gt = 0
    covered_base = 0
    covered_final = 0
    top1_counter: Dict[int, int] = {}
    rank_counter: Dict[int, int] = {}
    focus_top1_counter: Dict[int, int] = {pred_id: 0 for pred_id in focus_ids}
    margin_ab: List[float] = []
    examples = []

    with torch.no_grad():
        for batch_idx, (images, targets, metas) in enumerate(dataloaders["test"]):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = images.to(trainer.device)
            targets = trainer._move_targets(targets)
            features = model._extract_features(images)
            proposals, base_pair_idxs, final_pair_idxs, relation_logits = _prepare_relation_forward(
                model, relation_head, features, images, targets
            )

            for proposal, target, meta, base_pair_idx, final_pair_idx, rel_logit in zip(
                proposals, targets, metas, base_pair_idxs, final_pair_idxs, relation_logits
            ):
                gt_rels = target.get_field("relation_triplets").long() if target.has_field("relation_triplets") else torch.zeros((0, 3), dtype=torch.long, device=target.bbox.device)
                gt_labels = target.get_field("labels").long() if target.has_field("labels") else torch.zeros((0,), dtype=torch.long, device=target.bbox.device)
                for subj, obj, pred in gt_rels.tolist():
                    if int(pred) != gt_pred_id:
                        continue
                    total_gt += 1
                    if _pair_mask(base_pair_idx, subj, obj).any():
                        covered_base += 1
                    chosen = _best_pair_score(final_pair_idx, rel_logit, subj, obj)
                    if chosen is None:
                        continue
                    covered_final += 1
                    probs = chosen.softmax(dim=0)
                    order = torch.argsort(chosen[1:], descending=True) + 1
                    top1 = int(order[0].item())
                    top1_counter[top1] = top1_counter.get(top1, 0) + 1
                    if top1 in focus_top1_counter:
                        focus_top1_counter[top1] += 1
                    rank = int((order == gt_pred_id).nonzero(as_tuple=False)[0].item()) + 1
                    rank_counter[rank] = rank_counter.get(rank, 0) + 1

                    if len(focus_ids) >= 2:
                        margin_ab.append(float(chosen[focus_ids[0]].item() - chosen[focus_ids[1]].item()))

                    if len(examples) < args.max_examples:
                        top_ids = order[: max(int(args.topk), 1)].tolist()
                        top_entries = []
                        for pred_id in top_ids:
                            top_entries.append(
                                {
                                    "pred_id": int(pred_id),
                                    "pred_name": _class_name(rel_names, int(pred_id)),
                                    "logit": float(chosen[int(pred_id)].item()),
                                    "prob": float(probs[int(pred_id)].item()),
                                }
                            )
                        focus_entries = []
                        for pred_id in focus_ids:
                            focus_entries.append(
                                {
                                    "pred_id": int(pred_id),
                                    "pred_name": _class_name(rel_names, int(pred_id)),
                                    "logit": float(chosen[int(pred_id)].item()),
                                    "prob": float(probs[int(pred_id)].item()),
                                }
                            )
                        examples.append(
                            {
                                "image_id": int(meta.get("image_id", -1)),
                                "subj": int(subj),
                                "obj": int(obj),
                                "subj_label": int(gt_labels[int(subj)].item()) if int(subj) < len(gt_labels) else -1,
                                "obj_label": int(gt_labels[int(obj)].item()) if int(obj) < len(gt_labels) else -1,
                                "rank": rank,
                                "top_entries": top_entries,
                                "focus_entries": focus_entries,
                            }
                        )

    print(f"checkpoint: {args.checkpoint}")
    print(f"GT predicate: {args.gt_predicate} (id={gt_pred_id})")
    print(f"focus predicates: {[(pred_id, rel_names[pred_id]) for pred_id in focus_ids]}")
    print(f"total GT relations: {total_gt}")
    print(f"covered by base pairs: {covered_base}")
    print(f"covered by final pairs: {covered_final}")
    if total_gt > 0:
        print(f"base pair coverage ratio: {covered_base / total_gt:.6f}")
        print(f"final pair coverage ratio: {covered_final / total_gt:.6f}")

    print("\nTop-1 distribution on covered pairs:")
    for pred_id, count in sorted(top1_counter.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {rel_names[pred_id]} ({pred_id}): {count}")

    print("\nRank distribution of GT predicate on covered pairs:")
    for rank, count in sorted(rank_counter.items())[:10]:
        print(f"  rank {rank}: {count}")

    if margin_ab:
        mean_margin = sum(margin_ab) / len(margin_ab)
        pos = sum(1 for x in margin_ab if x > 0)
        print("\nMargin between first two focus predicates:")
        print(f"  {rel_names[focus_ids[0]]} - {rel_names[focus_ids[1]]} mean logit margin: {mean_margin:.6f}")
        print(f"  positive margin count: {pos} / {len(margin_ab)} = {pos / len(margin_ab):.6f}")

    print("\nFocus top-1 counts:")
    for pred_id in focus_ids:
        print(f"  {rel_names[pred_id]} ({pred_id}): {focus_top1_counter.get(pred_id, 0)}")

    if examples:
        print("\nExamples:")
        for ex in examples:
            print(
                f"  image_id={ex['image_id']} pair=({ex['subj']},{ex['obj']}) "
                f"{_class_name(class_names, ex['subj_label'])}->{_class_name(class_names, ex['obj_label'])} "
                f"gt_rank={ex['rank']}"
            )
            print(f"    topk={ex['top_entries']}")
            print(f"    focus={ex['focus_entries']}")


if __name__ == "__main__":
    main()
