from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
from sgg.data.statistics import predicate_histogram
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.structures.boxlist import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


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


def apply_runtime_cfg(cfg: Dict):
    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def _sync_cfg_from_train_metadata(cfg: Dict, datasets: Dict[str, object]) -> None:
    meta_dataset = datasets.get("train") or next(iter(datasets.values()))
    meta = meta_dataset.metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        meta.categories[i] for i in sorted(meta.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
        meta.predicates[i] for i in sorted(meta.predicates.keys())
    ]
    if not cfg["MODEL"]["ROI_RELATION_HEAD"].get("PREDICATE_COUNTS") and "train" in datasets:
        cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICATE_COUNTS"] = predicate_histogram(datasets["train"])


def _to_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu")


def _field(boxlist: BoxList, name: str, default: Optional[torch.Tensor] = None) -> torch.Tensor:
    if boxlist.has_field(name):
        return boxlist.get_field(name)
    if default is not None:
        return default
    raise KeyError(f"BoxList is missing required field {name!r}; fields={boxlist.fields()}")


def _labels_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.numel() == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=logits.device),
            torch.zeros((0,), dtype=torch.float32, device=logits.device),
        )
    if logits.size(1) <= 1:
        return (
            torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device),
            torch.ones((logits.size(0),), dtype=torch.float32, device=logits.device),
        )
    probs = logits.softmax(dim=1)
    scores, labels = probs[:, 1:].max(dim=1)
    return labels.long() + 1, scores.float()


def _class_name(class_names: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(class_names):
        return str(class_names[idx])
    return str(idx)


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


class ObjectDiagnostics:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.gt_count = torch.zeros(self.num_classes, dtype=torch.long)
        self.detector_correct = torch.zeros(self.num_classes, dtype=torch.long)
        self.refine_correct = torch.zeros(self.num_classes, dtype=torch.long)
        self.detector_confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)
        self.refine_confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

        self.det_retained_by_class = torch.zeros(self.num_classes, dtype=torch.long)
        self.refine_retained_by_class = torch.zeros(self.num_classes, dtype=torch.long)

        self.sgdet_gt_box_recalled = torch.zeros(self.num_classes, dtype=torch.long)
        self.sgdet_gt_detector_class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        self.sgdet_gt_refine_class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        self.sgdet_matched_detection_count = 0
        self.sgdet_matched_detector_correct = 0
        self.sgdet_matched_refine_correct = 0

        self.triplet_total = 0
        self.triplet_hit = 0
        self.triplet_missed = 0
        self.triplet_miss_detector_object_error = 0
        self.triplet_miss_refine_object_error = 0
        self.triplet_miss_box_error = 0
        self.triplet_miss_pair_error = 0
        self.triplet_miss_predicate_error = 0

    def update_gt_box_classification(
        self,
        gt_labels: torch.Tensor,
        detector_labels: torch.Tensor,
        refine_labels: torch.Tensor,
    ) -> None:
        gt_labels = gt_labels.long().cpu()
        detector_labels = detector_labels.long().cpu().clamp(min=0, max=self.num_classes - 1)
        refine_labels = refine_labels.long().cpu().clamp(min=0, max=self.num_classes - 1)
        for gt, det, ref in zip(gt_labels.tolist(), detector_labels.tolist(), refine_labels.tolist()):
            if not 0 <= gt < self.num_classes:
                continue
            self.gt_count[gt] += 1
            self.detector_correct[gt] += int(det == gt)
            self.refine_correct[gt] += int(ref == gt)
            self.detector_confusion[gt, det] += 1
            self.refine_confusion[gt, ref] += 1

    def update_retained_counts(self, detector_labels: torch.Tensor, refine_labels: torch.Tensor) -> None:
        for label in detector_labels.long().cpu().clamp(min=0, max=self.num_classes - 1).tolist():
            self.det_retained_by_class[label] += 1
        for label in refine_labels.long().cpu().clamp(min=0, max=self.num_classes - 1).tolist():
            self.refine_retained_by_class[label] += 1

    def update_sgdet_matching(
        self,
        target: BoxList,
        proposal: BoxList,
        detector_labels: torch.Tensor,
        refine_labels: torch.Tensor,
        iou_thresh: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return best detection index, IoU and validity for each GT object."""
        gt_labels = target.get_field("labels").long().cpu().clamp(min=0, max=self.num_classes - 1)
        for gt in gt_labels.tolist():
            self.gt_count[gt] += 1
        if len(target) == 0 or len(proposal) == 0:
            return (
                torch.full((len(target),), -1, dtype=torch.long),
                torch.zeros((len(target),), dtype=torch.float32),
                torch.zeros((len(target),), dtype=torch.bool),
            )
        ious = boxlist_iou(target, proposal, mode="obb" if target.mode == "xywha" and proposal.mode == "xywha" else "hbb").cpu()
        best_iou, best_det = ious.max(dim=1)
        valid = best_iou >= float(iou_thresh)
        detector_labels_cpu = detector_labels.long().cpu().clamp(min=0, max=self.num_classes - 1)
        refine_labels_cpu = refine_labels.long().cpu().clamp(min=0, max=self.num_classes - 1)
        for gt_idx, is_valid in enumerate(valid.tolist()):
            gt = int(gt_labels[gt_idx])
            if not is_valid:
                continue
            det_idx = int(best_det[gt_idx])
            self.sgdet_gt_box_recalled[gt] += 1
            self.sgdet_gt_detector_class_correct[gt] += int(detector_labels_cpu[det_idx] == gt)
            self.sgdet_gt_refine_class_correct[gt] += int(refine_labels_cpu[det_idx] == gt)

        det_best_iou, det_best_gt = ious.max(dim=0)
        det_valid = det_best_iou >= float(iou_thresh)
        for det_idx, is_valid in enumerate(det_valid.tolist()):
            if not is_valid:
                continue
            gt = int(gt_labels[int(det_best_gt[det_idx])])
            self.sgdet_matched_detection_count += 1
            self.sgdet_matched_detector_correct += int(detector_labels_cpu[det_idx] == gt)
            self.sgdet_matched_refine_correct += int(refine_labels_cpu[det_idx] == gt)
            self.detector_confusion[gt, int(detector_labels_cpu[det_idx])] += 1
            self.refine_confusion[gt, int(refine_labels_cpu[det_idx])] += 1
        return best_det, best_iou, valid

    def update_sgcls_triplet_failures(
        self,
        target: BoxList,
        pred: BoxList,
        detector_labels: torch.Tensor,
        refine_labels: torch.Tensor,
        topk: int,
    ) -> None:
        if not target.has_field("relation_triplets"):
            return
        rels = target.get_field("relation_triplets").long().cpu()
        if rels.numel() == 0:
            return
        pair_idx = _field(pred, "rel_pair_idxs", torch.zeros((0, 2), dtype=torch.long)).long().cpu()
        pred_rel = _field(pred, "pred_rel_labels", torch.zeros((0,), dtype=torch.long)).long().cpu()
        limit = min(int(topk), int(pair_idx.size(0)))
        pred_triples = {
            (int(pair_idx[i, 0]), int(pair_idx[i, 1]), int(pred_rel[i]))
            for i in range(limit)
        }
        pred_pairs = {
            (int(pair_idx[i, 0]), int(pair_idx[i, 1]))
            for i in range(limit)
        }
        gt_labels = target.get_field("labels").long().cpu()
        detector_labels = detector_labels.long().cpu()
        refine_labels = refine_labels.long().cpu()
        for subj, obj, pred_id in rels.tolist():
            self.triplet_total += 1
            key = (int(subj), int(obj), int(pred_id))
            pair = (int(subj), int(obj))
            refine_object_ok = (
                refine_labels[int(subj)] == gt_labels[int(subj)]
                and refine_labels[int(obj)] == gt_labels[int(obj)]
            )
            if key in pred_triples and bool(refine_object_ok):
                self.triplet_hit += 1
                continue
            self.triplet_missed += 1
            if detector_labels[int(subj)] != gt_labels[int(subj)] or detector_labels[int(obj)] != gt_labels[int(obj)]:
                self.triplet_miss_detector_object_error += 1
            if not bool(refine_object_ok):
                self.triplet_miss_refine_object_error += 1
            if pair not in pred_pairs:
                self.triplet_miss_pair_error += 1
            elif key not in pred_triples:
                self.triplet_miss_predicate_error += 1

    def update_sgdet_triplet_failures(
        self,
        target: BoxList,
        pred: BoxList,
        detector_labels: torch.Tensor,
        refine_labels: torch.Tensor,
        best_det: torch.Tensor,
        matched_valid: torch.Tensor,
        topk: int,
    ) -> None:
        if not target.has_field("relation_triplets"):
            return
        rels = target.get_field("relation_triplets").long().cpu()
        if rels.numel() == 0:
            return
        pair_idx = _field(pred, "rel_pair_idxs", torch.zeros((0, 2), dtype=torch.long)).long().cpu()
        pred_rel = _field(pred, "pred_rel_labels", torch.zeros((0,), dtype=torch.long)).long().cpu()
        limit = min(int(topk), int(pair_idx.size(0)))
        pair_to_predicates: Dict[Tuple[int, int], set[int]] = defaultdict(set)
        for i in range(limit):
            pair_to_predicates[(int(pair_idx[i, 0]), int(pair_idx[i, 1]))].add(int(pred_rel[i]))
        gt_labels = target.get_field("labels").long().cpu()
        detector_labels = detector_labels.long().cpu()
        refine_labels = refine_labels.long().cpu()
        for subj, obj, pred_id in rels.tolist():
            self.triplet_total += 1
            subj = int(subj)
            obj = int(obj)
            pred_id = int(pred_id)
            if subj >= matched_valid.numel() or obj >= matched_valid.numel() or not bool(matched_valid[subj]) or not bool(matched_valid[obj]):
                self.triplet_missed += 1
                self.triplet_miss_box_error += 1
                continue
            det_subj = int(best_det[subj])
            det_obj = int(best_det[obj])
            pair = (det_subj, det_obj)
            refine_object_ok = (
                refine_labels[det_subj] == gt_labels[subj]
                and refine_labels[det_obj] == gt_labels[obj]
            )
            if pred_id in pair_to_predicates.get(pair, set()) and bool(refine_object_ok):
                self.triplet_hit += 1
                continue
            self.triplet_missed += 1
            if detector_labels[det_subj] != gt_labels[subj] or detector_labels[det_obj] != gt_labels[obj]:
                self.triplet_miss_detector_object_error += 1
            if not bool(refine_object_ok):
                self.triplet_miss_refine_object_error += 1
            if pair not in pair_to_predicates:
                self.triplet_miss_pair_error += 1
            else:
                self.triplet_miss_predicate_error += 1


def _top_confusions(confusion: torch.Tensor, class_names: Sequence[str], topn: int) -> List[Dict]:
    rows = []
    num_classes = confusion.size(0)
    for gt in range(1, num_classes):
        for pred in range(0, num_classes):
            if gt == pred:
                continue
            count = int(confusion[gt, pred])
            if count <= 0:
                continue
            rows.append(
                {
                    "gt": gt,
                    "gt_name": _class_name(class_names, gt),
                    "pred": pred,
                    "pred_name": _class_name(class_names, pred),
                    "count": count,
                }
            )
    rows.sort(key=lambda item: item["count"], reverse=True)
    return rows[:topn]


def _per_class_summary(
    class_names: Sequence[str],
    gt_count: torch.Tensor,
    detector_correct: torch.Tensor,
    refine_correct: torch.Tensor,
    extra: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict]:
    rows = []
    extra = extra or {}
    for cls in range(1, gt_count.numel()):
        count = int(gt_count[cls])
        row = {
            "class": cls,
            "name": _class_name(class_names, cls),
            "count": count,
            "detector_recall": _safe_div(int(detector_correct[cls]), count),
            "refine_recall": _safe_div(int(refine_correct[cls]), count),
        }
        for key, tensor in extra.items():
            row[key] = int(tensor[cls])
        rows.append(row)
    rows.sort(key=lambda item: (item["refine_recall"], item["count"]), reverse=False)
    return rows


def _summary(diag: ObjectDiagnostics, class_names: Sequence[str], task: str, top_confusions: int) -> Dict:
    gt_total = int(diag.gt_count[1:].sum())
    det_correct = int(diag.detector_correct[1:].sum())
    ref_correct = int(diag.refine_correct[1:].sum())
    result = {
        "task": task,
        "object": {
            "gt_total": gt_total,
            "detector_accuracy": _safe_div(det_correct, gt_total),
            "refine_accuracy": _safe_div(ref_correct, gt_total),
            "detector_correct": det_correct,
            "refine_correct": ref_correct,
        },
        "triplet_failure": {
            "gt_triplets": int(diag.triplet_total),
            "hit_at_topk": int(diag.triplet_hit),
            "missed_at_topk": int(diag.triplet_missed),
            "miss_detector_object_error": int(diag.triplet_miss_detector_object_error),
            "miss_detector_object_error_ratio": _safe_div(
                diag.triplet_miss_detector_object_error, diag.triplet_missed
            ),
            "miss_refine_object_error": int(diag.triplet_miss_refine_object_error),
            "miss_refine_object_error_ratio": _safe_div(
                diag.triplet_miss_refine_object_error, diag.triplet_missed
            ),
            "miss_box_error": int(diag.triplet_miss_box_error),
            "miss_box_error_ratio": _safe_div(diag.triplet_miss_box_error, diag.triplet_missed),
            "miss_pair_error": int(diag.triplet_miss_pair_error),
            "miss_pair_error_ratio": _safe_div(diag.triplet_miss_pair_error, diag.triplet_missed),
            "miss_predicate_error": int(diag.triplet_miss_predicate_error),
            "miss_predicate_error_ratio": _safe_div(diag.triplet_miss_predicate_error, diag.triplet_missed),
        },
        "per_class": _per_class_summary(class_names, diag.gt_count, diag.detector_correct, diag.refine_correct),
        "retained_by_class": [
            {
                "class": cls,
                "name": _class_name(class_names, cls),
                "detector_retained": int(diag.det_retained_by_class[cls]),
                "refine_retained": int(diag.refine_retained_by_class[cls]),
            }
            for cls in range(1, diag.num_classes)
        ],
        "top_detector_confusions": _top_confusions(diag.detector_confusion, class_names, top_confusions),
        "top_refine_confusions": _top_confusions(diag.refine_confusion, class_names, top_confusions),
    }
    if task == "sgdet":
        box_recalled = int(diag.sgdet_gt_box_recalled[1:].sum())
        gt_det_cls = int(diag.sgdet_gt_detector_class_correct[1:].sum())
        gt_ref_cls = int(diag.sgdet_gt_refine_class_correct[1:].sum())
        result["sgdet"] = {
            "gt_object_count": gt_total,
            "gt_box_recall": _safe_div(box_recalled, gt_total),
            "gt_detector_class_recall": _safe_div(gt_det_cls, gt_total),
            "gt_refine_class_recall": _safe_div(gt_ref_cls, gt_total),
            "matched_detection_count": int(diag.sgdet_matched_detection_count),
            "matched_detector_accuracy": _safe_div(
                diag.sgdet_matched_detector_correct, diag.sgdet_matched_detection_count
            ),
            "matched_refine_accuracy": _safe_div(
                diag.sgdet_matched_refine_correct, diag.sgdet_matched_detection_count
            ),
            "per_class_detection": _per_class_summary(
                class_names,
                diag.gt_count,
                diag.sgdet_gt_detector_class_correct,
                diag.sgdet_gt_refine_class_correct,
                extra={"box_recalled": diag.sgdet_gt_box_recalled},
            ),
        }
    return result


@torch.no_grad()
def run_diagnostics(args) -> Dict:
    cfg = get_default_cfg()
    if args.config:
        cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)
    cfg["SOLVER"]["VAL_SPLIT"] = args.split
    cfg["DATALOADER"][f"{args.split.upper()}_BATCH_SIZE"] = args.batch_size
    cfg["TEST"]["IMS_PER_BATCH"] = args.batch_size

    splits = ["train", args.split] if args.split != "train" else ["train"]
    datasets = build_datasets(cfg, splits=splits)
    _sync_cfg_from_train_metadata(cfg, datasets)
    loaders = build_dataloaders(
        cfg,
        splits=(args.split,),
        datasets={args.split: datasets[args.split]},
        shuffle_map={args.split: False},
    )
    loader = loaders[args.split]
    class_names = list(cfg["MODEL"]["ROI_BOX_HEAD"].get("CLASS_NAMES", []))
    num_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
    task = str(cfg["MODEL"].get("TASK", "")).lower()
    if task not in {"sgcls", "sgdet"}:
        raise ValueError(f"This diagnostic is intended for sgcls/sgdet, got MODEL.TASK={task!r}")

    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=args.device, dataloaders={})
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    model = trainer.model
    model.eval()

    diag = ObjectDiagnostics(num_classes=num_classes)
    processed = 0
    iou_thresh = float(args.iou_thresh if args.iou_thresh is not None else cfg["TEST"].get("IOU_THRESHOLD", 0.5))
    topk = int(args.topk)

    for images, targets, metas in tqdm(loader, desc=f"ObjectDiag[{task}:{args.split}]"):
        if args.max_images > 0 and processed >= args.max_images:
            break
        images = images.to(args.device)
        targets = [target.to(args.device) for target in targets]
        detector_images, detector_targets = trainer._sgdet_detector_inputs_from_metas(metas)

        features = model._extract_features(images)
        proposals = model._prepare_eval_proposals(
            features,
            images,
            targets,
            detector_images=detector_images,
            detector_targets=detector_targets,
        )

        detector_labels_by_image = []
        detector_logits_by_image = []
        for proposal in proposals:
            logits = _field(
                proposal,
                "predict_logits",
                proposal.bbox.new_zeros((len(proposal), num_classes)),
            )
            labels = _field(proposal, "pred_labels", _labels_from_logits(logits)[0])
            detector_labels_by_image.append(_to_cpu_tensor(labels.long()))
            detector_logits_by_image.append(_to_cpu_tensor(logits.float()))

        _, preds, _ = model.roi_heads.relation(
            features,
            proposals,
            targets if task != "sgdet" else None,
            OBj=model,
        )

        for target, proposal, pred, detector_labels in zip(targets, proposals, preds, detector_labels_by_image):
            gt_labels = target.get_field("labels").long().cpu().clamp(min=0, max=num_classes - 1)
            refine_logits = _field(pred, "predict_logits", pred.bbox.new_zeros((len(pred), num_classes))).float()
            refine_labels = _field(pred, "pred_labels", _labels_from_logits(refine_logits)[0]).long().cpu()

            if len(detector_labels) != len(refine_labels):
                min_len = min(len(detector_labels), len(refine_labels))
                detector_labels = detector_labels[:min_len]
                refine_labels = refine_labels[:min_len]

            diag.update_retained_counts(detector_labels, refine_labels)
            if task == "sgcls":
                diag.update_gt_box_classification(gt_labels, detector_labels, refine_labels)
                diag.update_sgcls_triplet_failures(target.to("cpu"), pred.to("cpu"), detector_labels, refine_labels, topk)
            else:
                best_det, _, matched_valid = diag.update_sgdet_matching(
                    target.to("cpu"),
                    proposal.to("cpu"),
                    detector_labels,
                    refine_labels,
                    iou_thresh=iou_thresh,
                )
                diag.update_sgdet_triplet_failures(
                    target.to("cpu"),
                    pred.to("cpu"),
                    detector_labels,
                    refine_labels,
                    best_det,
                    matched_valid,
                    topk,
                )
            processed += 1
            if args.max_images > 0 and processed >= args.max_images:
                break

    result = _summary(diag, class_names, task=task, top_confusions=args.top_confusions)
    result["meta"] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "device": args.device,
        "images": processed,
        "topk": topk,
        "iou_thresh": iou_thresh,
        "filter_method": cfg["MODEL"]["ROI_RELATION_HEAD"].get("TEST_FILTER_METHOD"),
    }
    return result


def print_summary(result: Dict) -> None:
    obj = result["object"]
    print(
        "Object accuracy:",
        {
            "detector": round(float(obj["detector_accuracy"]), 4),
            "refine": round(float(obj["refine_accuracy"]), 4),
            "gt_total": int(obj["gt_total"]),
        },
        flush=True,
    )
    if "sgdet" in result:
        print(
            "Sgdet detection/classification:",
            {
                "box_recall": round(float(result["sgdet"]["gt_box_recall"]), 4),
                "gt_detector_class_recall": round(float(result["sgdet"]["gt_detector_class_recall"]), 4),
                "gt_refine_class_recall": round(float(result["sgdet"]["gt_refine_class_recall"]), 4),
                "matched_detector_acc": round(float(result["sgdet"]["matched_detector_accuracy"]), 4),
                "matched_refine_acc": round(float(result["sgdet"]["matched_refine_accuracy"]), 4),
            },
            flush=True,
        )
    print(
        "Triplet miss decomposition:",
        {
            key: round(float(value), 4) if "ratio" in key else int(value)
            for key, value in result["triplet_failure"].items()
        },
        flush=True,
    )
    print("Worst object classes by refine recall:", flush=True)
    for row in result["per_class"][:12]:
        if int(row["count"]) <= 0:
            continue
        print(
            f"  {row['class']:>2d} {row['name']:<32s} "
            f"count={row['count']:>6d} "
            f"det={row['detector_recall']:.4f} "
            f"ref={row['refine_recall']:.4f}",
            flush=True,
        )
    print("Top detector confusions:", flush=True)
    for row in result["top_detector_confusions"][:10]:
        print(
            f"  {row['gt_name']} -> {row['pred_name']}: {row['count']}",
            flush=True,
        )
    print("Top refine confusions:", flush=True)
    for row in result["top_refine_confusions"][:10]:
        print(
            f"  {row['gt_name']} -> {row['pred_name']}: {row['count']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sgcls/sgdet object classification bottlenecks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0, help="0 means full split.")
    parser.add_argument("--topk", type=int, default=2000)
    parser.add_argument("--iou-thresh", type=float, default=None)
    parser.add_argument("--top-confusions", type=int, default=50)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = run_diagnostics(args)
    output = args.output
    if not output:
        output_dir = Path(result["meta"]["checkpoint"]).parent if result["meta"]["checkpoint"] else Path("outputs/object_diagnostics")
        output = str(output_dir / f"{result['meta']['split']}_object_diagnostics.json")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print_summary(result)
    print(f"Wrote diagnostics: {output_path}", flush=True)


if __name__ == "__main__":
    main()
