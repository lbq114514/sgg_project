from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_right
import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
from pathlib import Path

from sgg.data.build import build_dataloaders
from sgg.evaluation import evaluate_sgg
from sgg.modeling.roi_heads.typed_hyper_rpcm import build_family_predicates
from sgg.modeling.detectors.class_channel_order import (
    BACKGROUND_LAST,
    INTERNAL_DETECTOR_CLASS_ORDER,
    is_detector_classifier_key,
    reorder_detector_classifier_rows,
)
from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


def _disable_tqdm_for_non_tty() -> bool:
    return not sys.stderr.isatty()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _predicate_name(predicate_names, idx: int) -> str:
    if predicate_names is None:
        return str(idx)
    if isinstance(predicate_names, dict):
        value = predicate_names.get(idx)
        return str(value) if value is not None else str(idx)
    if 0 <= idx < len(predicate_names):
        return str(predicate_names[idx])
    return str(idx)


def _format_metric_dict_row(label: str, values: Dict[int, float], topk: Sequence[int], label_width: int = 8) -> str:
    cells = [f"{label:<{label_width}}"]
    for k in topk:
        cells.append(f"@{int(k):<4} {float(values.get(int(k), 0.0)):.4f}")
    return " | ".join(cells)


def _format_optional_metric_dict_row(
    label: str,
    values: Dict[int, float],
    topk: Sequence[int],
    label_width: int = 8,
) -> str:
    cells = [f"{label:<{label_width}}"]
    for k in topk:
        if int(k) in values:
            cells.append(f"@{int(k):<4} {float(values[int(k)]):.4f}")
        else:
            cells.append(f"@{int(k):<4} {'n/a':>6}")
    return " | ".join(cells)


def _build_eval_report_lines(
    metrics: Dict,
    per_predicate_recall: Dict[int, Dict[int, float]],
    predicate_counts: Dict[int, int],
    predicate_names=None,
) -> List[str]:
    lines: List[str] = []
    topk = sorted(int(k) for k in metrics.get("R", {}).keys())
    if not topk:
        return lines

    lines.append(_format_metric_dict_row("R", metrics.get("R", {}), topk))
    lines.append(_format_metric_dict_row("mR", metrics.get("mR", {}), topk))
    lines.append(_format_metric_dict_row("HMR", metrics.get("HR", {}), topk))

    predicate_ids = sorted(
        {
            int(predicate)
            for recall_by_k in per_predicate_recall.values()
            for predicate in recall_by_k.keys()
        }
        | {int(predicate) for predicate in predicate_counts.keys()}
    )
    if predicate_ids:
        lines.append("Per-Relation Recall:")
        name_width = max(
            len("predicate"),
            max(len(_predicate_name(predicate_names, predicate)) for predicate in predicate_ids),
        )
        count_width = max(
            len("count"),
            max(len(str(int(predicate_counts.get(predicate, 0)))) for predicate in predicate_ids),
        )
        for predicate in predicate_ids:
            cells = [
                f"{_predicate_name(predicate_names, predicate):<{name_width}}",
                f"count {int(predicate_counts.get(predicate, 0)):>{count_width}}",
            ]
            for k in topk:
                recall_value = float(per_predicate_recall.get(int(k), {}).get(int(predicate), 0.0))
                cells.append(f"@{int(k):<4} {recall_value:.4f}")
            lines.append(" | ".join(cells))

    lines.append("Pair Accuracy:")
    lines.append(_format_optional_metric_dict_row("A", metrics.get("A", {}), topk))
    return lines


def _clip_grad_norm(
    named_parameters,
    max_norm: float,
    clip: bool = True,
    verbose: bool = False,
) -> float:
    max_norm = float(max_norm)
    total_norm_sq = 0.0
    param_to_norm: List[Tuple[str, float, Tuple[int, ...]]] = []
    params = []
    for name, param in named_parameters:
        if param.grad is None:
            continue
        grad_norm = float(param.grad.data.norm(2).item())
        total_norm_sq += grad_norm ** 2
        param_to_norm.append((name, grad_norm, tuple(param.shape)))
        params.append(param)

    total_norm = total_norm_sq ** 0.5
    if clip and max_norm > 0.0 and params:
        clip_grad_norm_(params, max_norm=max_norm)

    if verbose and param_to_norm:
        clip_coef = max_norm / (total_norm + 1e-6) if max_norm > 0.0 else 1.0
        print(f"GradNorm total={total_norm:.5f} clip_coef={clip_coef:.5f}", flush=True)
        for name, grad_norm, shape in sorted(param_to_norm, key=lambda item: -item[1]):
            print(f"  {name}: {grad_norm:.5f} shape={shape}", flush=True)

    return total_norm


def _build_eval_debug_lines(metrics: Dict, predicate_names=None, top_predicates: int = 10, top_images: int = 10) -> List[str]:
    lines: List[str] = []
    debug = metrics.get("debug")
    if not isinstance(debug, dict):
        return lines

    best_k = int(debug.get("best_k", 0))

    images = debug.get("hardest_images", [])
    if images:
        lines.append(f"Hardest Images @R{best_k}:")
        for item in images[: max(int(top_images), 0)]:
            lines.append(
                "  "
                f"image_id={int(item.get('image_id', -1))}, "
                f"gt_rel={int(item.get('gt_rel_count', 0))}, "
                f"gt_pair={int(item.get('gt_pair_count', 0))}, "
                f"pred_pair={int(item.get('pred_pair_count', 0))}, "
                f"triplet_R={float(item.get('triplet_recall', 0.0)):.3f}"
            )
    predicate_candidates = debug.get("predicate_candidate_coverage", [])
    if predicate_candidates:
        lines.append("Predicate Candidate Coverage:")
        for item in predicate_candidates[: max(int(top_predicates), 0)]:
            predicate = int(item.get("predicate", 0))
            lines.append(
                "  "
                f"{_predicate_name(predicate_names, predicate)} "
                f"count={int(item.get('count', 0))}, "
                f"recall={float(item.get('recall', 0.0)):.4f}, "
                f"semantic={float(item.get('semantic', 0.0)):.4f}, "
                f"ppn={float(item.get('ppn', 0.0)):.4f}, "
                f"degree={float(item.get('degree_cap', 0.0)):.4f}, "
                f"final={float(item.get('final', 0.0)):.4f}"
            )
    predicate_confusion = debug.get("predicate_pair_cls_confusion", [])
    if predicate_confusion:
        lines.append("GT Pair Top-Predicate Confusion:")
        for item in predicate_confusion[: max(int(top_predicates), 0)]:
            predicate = int(item.get("predicate", 0))
            predictions = item.get("predictions", [])
            pieces = []
            for pred_item in predictions:
                pred_id = int(pred_item.get("predicate", 0))
                pieces.append(
                    f"{_predicate_name(predicate_names, pred_id)}:{int(pred_item.get('count', 0))}"
                )
            lines.append(
                "  "
                f"{_predicate_name(predicate_names, predicate)} "
                f"covered={int(item.get('covered', 0))} -> "
                + ", ".join(pieces)
            )
    vehicle_aux = debug.get("vehicle_aux", [])
    if vehicle_aux:
        lines.append("Vehicle Aux:")
        for item in vehicle_aux[: max(int(top_predicates), 0)]:
            predicate = int(item.get("predicate", 0))
            lines.append(
                "  "
                f"{_predicate_name(predicate_names, predicate)} "
                f"pos={int(item.get('pos', 0))}, "
                f"aux_recall@0={float(item.get('aux_recall_at_0', 0.0)):.4f}, "
                f"avg_logit={float(item.get('avg_logit', 0.0)):.4f}"
            )
    return lines


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        self.milestones = list(milestones)
        self.gamma = float(gamma)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = str(warmup_method)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / float(max(self.warmup_iters, 1))
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        gamma=0.5,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        patience=2,
        threshold=1e-4,
        cooldown=1,
    ):
        self.optimizer = optimizer
        self.gamma = float(gamma)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = str(warmup_method)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.stage_count = 0
        self.best = -1e12
        self.num_bad_epochs = 0
        self.under_cooldown = self.cooldown
        self.last_epoch = 0

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / float(max(self.warmup_iters, 1))
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** self.stage_count for base_lr in self.base_lrs]

    def step(self, metrics=None, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        if metrics is not None:
            metrics = float(metrics)
            if metrics > (self.best + self.threshold):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            if self.under_cooldown > 0:
                self.under_cooldown -= 1
                self.num_bad_epochs = 0
            if self.num_bad_epochs >= self.patience:
                self.stage_count += 1
                self.under_cooldown = self.cooldown
                self.num_bad_epochs = 0
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr_ratio=0.0,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        self.max_iters = max(int(max_iters), 1)
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = str(warmup_method)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / float(max(self.warmup_iters, 1))
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress_denom = max(self.max_iters - self.warmup_iters, 1)
        progress = min(max(self.last_epoch - self.warmup_iters, 0), progress_denom) / float(progress_denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupLinearDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr_ratio=0.0,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        self.max_iters = max(int(max_iters), 1)
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = str(warmup_method)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / float(max(self.warmup_iters, 1))
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress_denom = max(self.max_iters - self.warmup_iters, 1)
        progress = min(max(self.last_epoch - self.warmup_iters, 0), progress_denom) / float(progress_denom)
        factor = 1.0 - (1.0 - self.min_lr_ratio) * progress
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        gamma=0.999,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        self.gamma = float(gamma)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        self.warmup_method = str(warmup_method)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / float(max(self.warmup_iters, 1))
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        decay_step = max(self.last_epoch - self.warmup_iters, 0)
        factor = self.gamma ** decay_step
        return [base_lr * factor for base_lr in self.base_lrs]


class NoOpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list(self.base_lrs)


def _shift_boxes_to_source(boxes: torch.Tensor, mode: str, origin: Tuple[int, int], scale: Tuple[float, float]) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    x0, y0 = float(origin[0]), float(origin[1])
    sx = max(float(scale[0]), 1e-6)
    sy = max(float(scale[1]), 1e-6)
    shifted = boxes.clone()
    if mode == "xywha":
        shifted[:, 0] = shifted[:, 0] / sx + x0
        shifted[:, 1] = shifted[:, 1] / sy + y0
        shifted[:, 2] = shifted[:, 2] / sx
        shifted[:, 3] = shifted[:, 3] / sy
    else:
        shifted[:, [0, 2]] = shifted[:, [0, 2]] / sx + x0
        shifted[:, [1, 3]] = shifted[:, [1, 3]] / sy + y0
    return shifted


def _tile_key(meta: Dict) -> int:
    return int(meta.get("source_image_id", meta.get("image_id", 0)))


def _tile_size(meta: Dict) -> Tuple[int, int]:
    return (
        int(meta.get("source_width", meta.get("width", 1))),
        int(meta.get("source_height", meta.get("height", 1))),
    )


def _remap_relation_triplets(rel_triplets: torch.Tensor, index_map: Sequence[int]) -> torch.Tensor:
    if rel_triplets.numel() == 0:
        return rel_triplets.new_zeros((0, 3))
    remapped = []
    for subj, obj, pred in rel_triplets.tolist():
        gsub = index_map[int(subj)]
        gobj = index_map[int(obj)]
        if gsub != gobj:
            remapped.append((gsub, gobj, int(pred)))
    if not remapped:
        return rel_triplets.new_zeros((0, 3))
    remapped = sorted(set(remapped))
    return torch.tensor(remapped, dtype=torch.long)


def _pair_field(pred: BoxList) -> torch.Tensor:
    for name in ("rel_pair_idx", "rel_pair_idxs", "pair_idx"):
        if pred.has_field(name):
            return pred.get_field(name).long()
    return torch.zeros((0, 2), dtype=torch.long)


def _rel_score_field(pred: BoxList) -> torch.Tensor:
    for name in ("pred_rel_scores", "rel_scores", "rel_logits"):
        if pred.has_field(name):
            value = pred.get_field(name).float()
            if name == "rel_logits":
                value = value.softmax(-1)
            return value
    return torch.zeros((0, 0), dtype=torch.float32)


def _merge_object_boxlists(
    boxlists: Sequence[BoxList],
    metas: Sequence[Dict],
    label_field: str,
    score_field: Optional[str],
    iou_thresh: float,
) -> Tuple[BoxList, List[List[int]]]:
    if not boxlists:
        raise ValueError("Expected at least one BoxList to merge.")
    mode = boxlists[0].mode
    size = _tile_size(metas[0])
    merged_boxes: List[torch.Tensor] = []
    merged_labels: List[int] = []
    merged_scores: List[float] = []
    local_maps: List[List[int]] = []

    for boxlist, meta in zip(boxlists, metas):
        boxes = _shift_boxes_to_source(
            boxlist.bbox.detach().cpu(),
            boxlist.mode,
            meta.get("tile_origin", (0, 0)),
            meta.get("tile_scale", (1.0, 1.0)),
        )
        labels = boxlist.get_field(label_field).detach().cpu().long()
        if score_field is not None and boxlist.has_field(score_field):
            scores = boxlist.get_field(score_field).detach().cpu().float()
        else:
            scores = torch.ones((len(boxlist),), dtype=torch.float32)

        idx_map: List[int] = []
        for idx in range(boxes.size(0)):
            label = int(labels[idx].item())
            score = float(scores[idx].item())
            match = -1
            best_iou = -1.0
            for global_idx, global_label in enumerate(merged_labels):
                if global_label != label:
                    continue
                cur = BoxList(boxes[idx : idx + 1], size, mode)
                ref = BoxList(merged_boxes[global_idx].unsqueeze(0), size, mode)
                iou = float(boxlist_iou(cur, ref, mode="auto")[0, 0].item())
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    match = global_idx
            if match >= 0:
                idx_map.append(match)
                if score >= merged_scores[match]:
                    merged_boxes[match] = boxes[idx]
                    merged_scores[match] = score
            else:
                idx_map.append(len(merged_boxes))
                merged_boxes.append(boxes[idx])
                merged_labels.append(label)
                merged_scores.append(score)
        local_maps.append(idx_map)

    if merged_boxes:
        bbox = torch.stack(merged_boxes, dim=0)
        label_tensor = torch.tensor(merged_labels, dtype=torch.long)
        score_tensor = torch.tensor(merged_scores, dtype=torch.float32)
    else:
        box_dim = 5 if mode == "xywha" else 4
        bbox = torch.zeros((0, box_dim), dtype=torch.float32)
        label_tensor = torch.zeros((0,), dtype=torch.long)
        score_tensor = torch.zeros((0,), dtype=torch.float32)

    merged = BoxList(bbox, size, mode)
    merged.add_field(label_field, label_tensor)
    if score_field is not None:
        merged.add_field(score_field, score_tensor)
    return merged, local_maps


def _merge_targets(boxlists: Sequence[BoxList], metas: Sequence[Dict], iou_thresh: float) -> Tuple[BoxList, List[List[int]]]:
    merged, local_maps = _merge_object_boxlists(boxlists, metas, label_field="labels", score_field=None, iou_thresh=iou_thresh)
    relation_triplets = []
    for boxlist, idx_map in zip(boxlists, local_maps):
        if boxlist.has_field("relation_triplets"):
            relation_triplets.append(_remap_relation_triplets(boxlist.get_field("relation_triplets").long().cpu(), idx_map))
    if relation_triplets:
        rel_triplets = torch.cat(relation_triplets, dim=0) if relation_triplets else torch.zeros((0, 3), dtype=torch.long)
        if rel_triplets.numel() > 0:
            rel_triplets = torch.unique(rel_triplets, dim=0)
    else:
        rel_triplets = torch.zeros((0, 3), dtype=torch.long)
    pair_labels = torch.zeros((len(merged), len(merged)), dtype=torch.long)
    for subj, obj, pred in rel_triplets.tolist():
        if subj != obj:
            pair_labels[subj, obj] = int(pred)
    merged.add_field("relation_triplets", rel_triplets)
    merged.add_field("pair_labels", pair_labels)
    if boxlists and boxlists[0].has_field("image_id"):
        merged.add_field("image_id", torch.tensor([_tile_key(metas[0])], dtype=torch.long))
    return merged, local_maps


def _merge_predictions(
    boxlists: Sequence[BoxList],
    metas: Sequence[Dict],
    iou_thresh: float,
    task: str,
    canonical_targets: Optional[BoxList] = None,
    canonical_maps: Optional[Sequence[Sequence[int]]] = None,
) -> BoxList:
    if canonical_targets is not None and canonical_maps is not None:
        merged = BoxList(canonical_targets.bbox.clone(), canonical_targets.size, canonical_targets.mode)
        merged_labels = torch.zeros((len(merged),), dtype=torch.long)
        merged_scores = torch.zeros((len(merged),), dtype=torch.float32)
        local_maps = [list(idx_map) for idx_map in canonical_maps]
    else:
        merged, local_maps = _merge_object_boxlists(
            boxlists,
            metas,
            label_field="pred_labels" if boxlists and boxlists[0].has_field("pred_labels") else "labels",
            score_field="pred_scores" if boxlists and boxlists[0].has_field("pred_scores") else None,
            iou_thresh=iou_thresh,
        )
        merged_labels = merged.get_field("pred_labels") if merged.has_field("pred_labels") else merged.get_field("labels")
        merged_scores = merged.get_field("pred_scores") if merged.has_field("pred_scores") else torch.ones((len(merged),), dtype=torch.float32)

    rel_score_dim = 0
    pair_scores: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
    for boxlist, idx_map in zip(boxlists, local_maps):
        labels = None
        scores = None
        if boxlist.has_field("pred_labels"):
            labels = boxlist.get_field("pred_labels").detach().cpu().long()
        elif boxlist.has_field("labels"):
            labels = boxlist.get_field("labels").detach().cpu().long()
        if boxlist.has_field("pred_scores"):
            scores = boxlist.get_field("pred_scores").detach().cpu().float()
        else:
            scores = torch.ones((len(boxlist),), dtype=torch.float32)
        if labels is not None:
            for local_idx, global_idx in enumerate(idx_map):
                if scores[local_idx] >= merged_scores[global_idx]:
                    merged_labels[global_idx] = labels[local_idx]
                    merged_scores[global_idx] = scores[local_idx]

        pair_idx = _pair_field(boxlist)
        rel_scores = _rel_score_field(boxlist)
        if rel_scores.numel() == 0 or pair_idx.numel() == 0:
            continue
        rel_scores = rel_scores.detach().cpu().float()
        pair_idx = pair_idx.detach().cpu().long()
        rel_score_dim = max(rel_score_dim, rel_scores.size(1))
        for rel_id in range(pair_idx.size(0)):
            gsub = idx_map[int(pair_idx[rel_id, 0].item())]
            gobj = idx_map[int(pair_idx[rel_id, 1].item())]
            if gsub == gobj:
                continue
            key = (gsub, gobj)
            score_vec = rel_scores[rel_id]
            prev = pair_scores.get(key)
            if prev is None:
                pair_scores[key] = score_vec
            else:
                if prev.size(0) != score_vec.size(0):
                    max_dim = max(prev.size(0), score_vec.size(0))
                    prev_pad = torch.zeros((max_dim,), dtype=prev.dtype)
                    cur_pad = torch.zeros((max_dim,), dtype=score_vec.dtype)
                    prev_pad[: prev.size(0)] = prev
                    cur_pad[: score_vec.size(0)] = score_vec
                    prev = prev_pad
                    score_vec = cur_pad
                pair_scores[key] = torch.maximum(prev, score_vec)

    if canonical_targets is not None:
        merged.add_field("labels", canonical_targets.get_field("labels").clone())
    merged.add_field("pred_labels", merged_labels)
    merged.add_field("pred_scores", merged_scores)
    if pair_scores:
        pair_items = list(pair_scores.items())
        pair_idx = torch.tensor([list(key) for key, _ in pair_items], dtype=torch.long)
        rel_scores = torch.stack([value if value.size(0) == rel_score_dim else F.pad(value, (0, rel_score_dim - value.size(0))) for _, value in pair_items], dim=0)
        merged.add_field("rel_pair_idx", pair_idx)
        merged.add_field("rel_pair_idxs", pair_idx)
        merged.add_field("pred_rel_scores", rel_scores)
        merged.add_field("pred_rel_labels", rel_scores.argmax(dim=1))
    else:
        merged.add_field("rel_pair_idx", torch.zeros((0, 2), dtype=torch.long))
        merged.add_field("rel_pair_idxs", torch.zeros((0, 2), dtype=torch.long))
        merged.add_field("pred_rel_scores", torch.zeros((0, rel_score_dim), dtype=torch.float32))
        merged.add_field("pred_rel_labels", torch.zeros((0,), dtype=torch.long))
    if task == "predcls" and merged.has_field("labels"):
        merged.add_field("pred_labels", merged.get_field("labels").clone())
        merged.add_field("pred_scores", torch.ones((len(merged),), dtype=torch.float32))
    return merged


def _merge_tile_batches(
    predictions: Sequence[BoxList],
    targets: Sequence[BoxList],
    metas: Sequence[Dict],
    task: str,
    iou_thresh: float,
) -> Tuple[List[BoxList], List[BoxList]]:
    grouped: "OrderedDict[int, List[int]]" = OrderedDict()
    for idx, meta in enumerate(metas):
        grouped.setdefault(_tile_key(meta), []).append(idx)

    merged_preds: List[BoxList] = []
    merged_targets: List[BoxList] = []
    for _, indices in grouped.items():
        group_preds = [predictions[i] for i in indices]
        group_targets = [targets[i] for i in indices]
        group_metas = [metas[i] for i in indices]
        merged_target, target_maps = _merge_targets(group_targets, group_metas, iou_thresh)
        if task in {"predcls", "sgcls"}:
            merged_pred = _merge_predictions(
                group_preds,
                group_metas,
                iou_thresh,
                task=task,
                canonical_targets=merged_target,
                canonical_maps=target_maps,
            )
        else:
            merged_pred = _merge_predictions(group_preds, group_metas, iou_thresh, task=task)
        merged_preds.append(merged_pred)
        merged_targets.append(merged_target)
    return merged_preds, merged_targets


class Trainer:
    def __init__(
        self,
        cfg: Dict,
        model: torch.nn.Module,
        device: str = "cpu",
        dataloaders: Optional[Dict[str, object]] = None,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.dataloaders = dataloaders or {}
        self._apply_freeze_config()
        self.optimizer = self._build_optimizer(model)
        self.global_step = 0
        self.warmup_iters = int(cfg["SOLVER"].get("WARMUP_ITERS", 0))
        self.warmup_epochs = float(cfg["SOLVER"].get("WARMUP_EPOCHS", 0))
        self.warmup_factor = float(cfg["SOLVER"].get("WARMUP_FACTOR", 0.001))
        self.warmup_method = str(cfg["SOLVER"].get("WARMUP_METHOD", "linear"))
        self.output_dir = Path(cfg["SOLVER"].get("OUTPUT_DIR", "outputs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_period = int(cfg["SOLVER"].get("CHECKPOINT_PERIOD", 1))
        self.val_period = int(cfg["SOLVER"].get("VAL_PERIOD", 1))
        self.val_start_period = int(cfg["SOLVER"].get("VAL_START_PERIOD", 0))
        self.print_grad_freq = int(cfg["SOLVER"].get("PRINT_GRAD_FREQ", 50))
        self.grad_norm_clip = float(cfg["SOLVER"].get("GRAD_NORM_CLIP", 0.0))
        self._scheduler_resolved = False
        self.scheduler = self._build_scheduler()
        self.best_metrics = {
            "R": float("-inf"),
            "mR": float("-inf"),
            "HR": float("-inf"),
        }

    def _apply_freeze_config(self):
        model_cfg = self.cfg.get("MODEL", {})
        sgdet_compat_freeze = (
            str(model_cfg.get("TASK", "")).lower() == "sgdet"
            and bool(model_cfg.get("SGDET_COMPAT", {}).get("ENABLED", False))
            and bool(model_cfg.get("SGDET_COMPAT", {}).get("FREEZE_DETECTOR", True))
        )
        freeze_map = {
            "backbone": bool(model_cfg.get("FREEZE_BACKBONE", False)) or sgdet_compat_freeze,
            "backbone_d2": bool(model_cfg.get("FREEZE_BACKBONE", False)) or sgdet_compat_freeze,
            "neck": bool(model_cfg.get("FREEZE_NECK", False)) or sgdet_compat_freeze,
            "neck_d2": bool(model_cfg.get("FREEZE_NECK", False)) or sgdet_compat_freeze,
            "rpn_head": bool(model_cfg.get("FREEZE_RPN_HEAD", False)) or sgdet_compat_freeze,
            "rpn_head_d2": bool(model_cfg.get("FREEZE_RPN_HEAD", False)) or sgdet_compat_freeze,
            "roi_head": bool(model_cfg.get("FREEZE_ROI_HEAD", False)) or sgdet_compat_freeze,
            "roi_head_d2": bool(model_cfg.get("FREEZE_ROI_HEAD", False)) or sgdet_compat_freeze,
            "det_tower": sgdet_compat_freeze,
            "rpn_objectness_head": sgdet_compat_freeze,
            "rpn_box_head": sgdet_compat_freeze,
        }
        for module_name, should_freeze in freeze_map.items():
            if not should_freeze or not hasattr(self.model, module_name):
                continue
            module = getattr(self.model, module_name)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
            num_params = sum(p.numel() for p in module.parameters())
            print(f"Frozen {module_name}: {num_params} parameters", flush=True)

    def _set_frozen_modules_eval(self):
        model_cfg = self.cfg.get("MODEL", {})
        sgdet_compat_freeze = (
            str(model_cfg.get("TASK", "")).lower() == "sgdet"
            and bool(model_cfg.get("SGDET_COMPAT", {}).get("ENABLED", False))
            and bool(model_cfg.get("SGDET_COMPAT", {}).get("FREEZE_DETECTOR", True))
        )
        for module_name, should_freeze in {
            "backbone": bool(model_cfg.get("FREEZE_BACKBONE", False)) or sgdet_compat_freeze,
            "backbone_d2": bool(model_cfg.get("FREEZE_BACKBONE", False)) or sgdet_compat_freeze,
            "neck": bool(model_cfg.get("FREEZE_NECK", False)) or sgdet_compat_freeze,
            "neck_d2": bool(model_cfg.get("FREEZE_NECK", False)) or sgdet_compat_freeze,
            "rpn_head": bool(model_cfg.get("FREEZE_RPN_HEAD", False)) or sgdet_compat_freeze,
            "rpn_head_d2": bool(model_cfg.get("FREEZE_RPN_HEAD", False)) or sgdet_compat_freeze,
            "roi_head": bool(model_cfg.get("FREEZE_ROI_HEAD", False)) or sgdet_compat_freeze,
            "roi_head_d2": bool(model_cfg.get("FREEZE_ROI_HEAD", False)) or sgdet_compat_freeze,
            "det_tower": sgdet_compat_freeze,
            "rpn_objectness_head": sgdet_compat_freeze,
            "rpn_box_head": sgdet_compat_freeze,
        }.items():
            if should_freeze and hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
                    # Match SGG-Toolkit's relation training: freeze detector
                    # parameters after model.train(), but keep module.training
                    # unchanged so detector train-mode branches still run.

    def _build_optimizer(self, model: torch.nn.Module):
        solver_cfg = self.cfg["SOLVER"]
        name = str(solver_cfg.get("OPTIMIZER", "SGD")).lower()
        base_lr = float(solver_cfg["BASE_LR"])
        bias_lr_factor = float(solver_cfg.get("BIAS_LR_FACTOR", 1.0))
        weight_decay = float(solver_cfg.get("WEIGHT_DECAY", 0.0))
        weight_decay_bias = float(solver_cfg.get("WEIGHT_DECAY_BIAS", weight_decay))
        momentum = float(solver_cfg.get("MOMENTUM", 0.9))
        betas = tuple(float(v) for v in solver_cfg.get("BETAS", (0.9, 0.999)))
        eps = float(solver_cfg.get("EPS", 1e-8))
        if bool(solver_cfg.get("LR_SCALE_BY_BATCH", True)):
            rl_factor = float(self.cfg.get("DATALOADER", {}).get("TRAIN_BATCH_SIZE", self.cfg.get("DATALOADER", {}).get("BATCH_SIZE", 1)))
        else:
            rl_factor = 1.0

        predictor_name = str(self.cfg.get("MODEL", {}).get("ROI_RELATION_HEAD", {}).get("PREDICTOR", ""))
        slow_heads = ["roi_heads.relation.box_feature_extractor", "roi_heads.relation.union_feature_extractor.feature_extractor"] if predictor_name == "IMPPredictor" else []
        slow_ratio = 10.0

        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = base_lr
            decay = weight_decay
            if "bias" in key:
                lr = base_lr * bias_lr_factor
                decay = weight_decay_bias
            if "norm" in key:
                decay = 0.0
            for item in slow_heads:
                if item in key:
                    lr = lr / slow_ratio
                    break
            params.append({"params": [value], "lr": lr * rl_factor, "weight_decay": decay})
        if not params:
            raise ValueError("No trainable parameters remain after applying freeze config.")
        scaled_lr = base_lr * rl_factor
        if name == "sgd":
            return SGD(params, lr=scaled_lr, momentum=momentum)
        if name == "adam":
            return Adam(params, lr=scaled_lr, betas=betas, eps=eps)
        if name == "adamw":
            return AdamW(params, lr=scaled_lr, betas=betas, eps=eps)
        raise ValueError(f"Unsupported optimizer: {solver_cfg.get('OPTIMIZER')}")

    def _resolve_scheduler_steps(self, num_iters_per_epoch: Optional[int] = None):
        solver_cfg = self.cfg["SOLVER"]
        schedule_cfg = solver_cfg.get("SCHEDULE", {})
        unit = str(schedule_cfg.get("UNIT", "iter")).lower()
        if unit not in {"iter", "epoch"}:
            raise ValueError(f"Unsupported scheduler unit: {unit}")
        if unit == "epoch":
            if num_iters_per_epoch is None:
                return None
            return [max(int(round(step * num_iters_per_epoch)), 1) for step in solver_cfg.get("STEPS", [])]
        return [int(step) for step in solver_cfg.get("STEPS", [])]

    def _build_scheduler(self, num_iters_per_epoch: Optional[int] = None):
        solver_cfg = self.cfg["SOLVER"]
        schedule_cfg = solver_cfg.get("SCHEDULE", {})
        schedule_type = str(schedule_cfg.get("TYPE", "WarmupMultiStepLR"))
        schedule_type_lower = schedule_type.lower()
        schedule_unit = str(schedule_cfg.get("UNIT", "iter")).lower()
        steps = self._resolve_scheduler_steps(num_iters_per_epoch)
        needs_resolution = steps is None or (self.warmup_iters <= 0 and self.warmup_epochs > 0 and num_iters_per_epoch is None)
        self._scheduler_resolved = not needs_resolution
        if schedule_type_lower in {"none", "constant", "noop"}:
            return NoOpLR(self.optimizer)
        if schedule_type == "WarmupReduceLROnPlateau":
            return WarmupReduceLROnPlateau(
                self.optimizer,
                gamma=float(schedule_cfg.get("FACTOR", 0.1)),
                warmup_factor=self.warmup_factor,
                warmup_iters=self.warmup_iters,
                warmup_method=self.warmup_method,
                patience=int(schedule_cfg.get("PATIENCE", 2)),
                threshold=float(schedule_cfg.get("THRESHOLD", 1e-4)),
                cooldown=int(schedule_cfg.get("COOLDOWN", 0)),
            )
        if num_iters_per_epoch is None:
            total_iters = max(int(solver_cfg.get("MAX_EPOCHS", 1)), 1)
        elif schedule_unit == "epoch":
            total_iters = max(int(round(float(solver_cfg.get("MAX_EPOCHS", 1)) * num_iters_per_epoch)), 1)
        else:
            total_iters = max(int(schedule_cfg.get("MAX_ITERS", 0)), 1)
            if total_iters <= 1:
                total_iters = max(int(round(float(solver_cfg.get("MAX_EPOCHS", 1)) * num_iters_per_epoch)), 1)
        if schedule_type == "WarmupCosineLR":
            return WarmupCosineLR(
                self.optimizer,
                max_iters=total_iters,
                min_lr_ratio=float(schedule_cfg.get("MIN_LR_RATIO", 0.0)),
                warmup_factor=self.warmup_factor,
                warmup_iters=self.warmup_iters,
                warmup_method=self.warmup_method,
            )
        if schedule_type == "WarmupLinearDecayLR":
            return WarmupLinearDecayLR(
                self.optimizer,
                max_iters=total_iters,
                min_lr_ratio=float(schedule_cfg.get("MIN_LR_RATIO", 0.0)),
                warmup_factor=self.warmup_factor,
                warmup_iters=self.warmup_iters,
                warmup_method=self.warmup_method,
            )
        if schedule_type == "WarmupExponentialLR":
            return WarmupExponentialLR(
                self.optimizer,
                gamma=float(schedule_cfg.get("EXP_GAMMA", schedule_cfg.get("GAMMA", 0.999))),
                warmup_factor=self.warmup_factor,
                warmup_iters=self.warmup_iters,
                warmup_method=self.warmup_method,
            )
        return WarmupMultiStepLR(
            self.optimizer,
            milestones=steps or [],
            gamma=float(solver_cfg.get("GAMMA", 0.1)),
            warmup_factor=self.warmup_factor,
            warmup_iters=self.warmup_iters,
            warmup_method=self.warmup_method,
        )

    def save_checkpoint(self, epoch: int, name: str = "model_last.pth", metrics: Optional[Dict] = None):
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else None,
            "global_step": self.global_step,
            "cfg": self.cfg,
            "detector_class_channel_order": INTERNAL_DETECTOR_CLASS_ORDER,
        }
        if metrics is not None:
            payload["metrics"] = metrics
        path = self.output_dir / name
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        checkpoint_class_order = (
            ckpt.get("detector_class_channel_order") if isinstance(ckpt, dict) else None
        )
        if checkpoint_class_order is None:
            print(
                "Checkpoint detector class-channel order: <unmarked>; "
                "loading unchanged as background_first for backward compatibility. "
                "Migrate explicitly before using an old checkpoint for sgcls/sgdet "
                "if its detector was initialized from the original background-last STAR OBB weights.",
                flush=True,
            )
        else:
            print(
                "Checkpoint detector class-channel order: "
                f"{checkpoint_class_order}",
                flush=True,
            )
        state_dict = self._checkpoint_state_dict(ckpt)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        filter_prefixes = (
            "roi_heads.relation.ppg.",
            "roi_heads.relation.ppn.",
        )
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(filter_prefixes)
        ]
        unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.startswith(filter_prefixes)
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is incompatible with the model outside the independent "
                f"pair filter: missing_keys={missing}, unexpected_keys={unexpected}"
            )
        ignored_missing = len(incompatible.missing_keys) - len(missing)
        ignored_unexpected = len(incompatible.unexpected_keys) - len(unexpected)
        if ignored_missing or ignored_unexpected:
            print(
                "Checkpoint pair-filter state ignored:",
                {"missing": ignored_missing, "unexpected": ignored_unexpected},
                flush=True,
            )
        if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if isinstance(ckpt, dict) and ckpt.get("scheduler") is not None and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if isinstance(ckpt, dict) and "global_step" in ckpt:
            self.global_step = int(ckpt["global_step"])
        return ckpt

    @staticmethod
    def _checkpoint_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "module"):
                value = ckpt.get(key)
                if isinstance(value, dict):
                    return value
        return ckpt

    @staticmethod
    def _legacy_rpcm_key_candidates(key: str) -> list[str]:
        candidates = [key]
        if ".patch_embed.projection." in key:
            candidates.append(key.replace(".patch_embed.projection.", ".patch_embed.proj."))
        if key.startswith("roi_heads.relation.PPG."):
            candidates.append("roi_heads.relation.ppg." + key[len("roi_heads.relation.PPG."):])
        if key.startswith("roi_heads.relation.union_feature_extractor.feature_extractor.fc6."):
            candidates.append(
                "roi_heads.relation.union_feature_extractor.head.1."
                + key[len("roi_heads.relation.union_feature_extractor.feature_extractor.fc6."):]
            )
        if key.startswith("roi_heads.relation.box_feature_extractor.fc6."):
            candidates.append(
                "roi_heads.relation.box_feature_extractor.head.1."
                + key[len("roi_heads.relation.box_feature_extractor.fc6."):]
            )
        if key.startswith("roi_heads.box.feature_extractor.fc6."):
            suffix = key[len("roi_heads.box.feature_extractor.fc6."):]
            candidates.extend(
                [
                    "roi_head.bbox_head.shared_fcs.0." + suffix,
                    "roi_head_d2.bbox_head.shared_fcs.0." + suffix,
                ]
            )
        if key.startswith("roi_heads.box.feature_extractor.fc7."):
            suffix = key[len("roi_heads.box.feature_extractor.fc7."):]
            candidates.extend(
                [
                    "roi_head.bbox_head.shared_fcs.1." + suffix,
                    "roi_head_d2.bbox_head.shared_fcs.1." + suffix,
                ]
            )
        if key.startswith("roi_heads.box.predictor.cls_score."):
            suffix = key[len("roi_heads.box.predictor.cls_score."):]
            candidates.extend(
                [
                    "roi_head.bbox_head.fc_cls." + suffix,
                    "roi_head_d2.bbox_head.fc_cls." + suffix,
                ]
            )
        return candidates

    def _load_legacy_rpcm_model_only(self, source: dict, path: str):
        target = self.model.state_dict()
        predictor_prefix = "roi_heads.relation.predictor."
        update = {}
        remapped = {}
        skipped_shape = []
        used_source_keys = set()
        reordered_detector_classifier = []

        for source_key, source_value in source.items():
            if source_key not in target:
                continue
            if not hasattr(source_value, "shape") or source_value.shape != target[source_key].shape:
                skipped_shape.append(
                    (
                        source_key,
                        source_key,
                        tuple(source_value.shape) if hasattr(source_value, "shape") else "<no-shape>",
                        tuple(target[source_key].shape),
                    )
                )
                continue
            value_to_load = source_value
            # A legacy RPCM file can already expose the mmrotate detector
            # classifier under its current key.  It is still background-last;
            # direct loading must not bypass the conversion applied to the
            # old ``roi_heads.box.predictor.cls_score`` alias below.
            if is_detector_classifier_key(source_key):
                value_to_load = reorder_detector_classifier_rows(
                    source_value,
                    source_order=BACKGROUND_LAST,
                    target_order=INTERNAL_DETECTOR_CLASS_ORDER,
                )
                reordered_detector_classifier.append((source_key, source_key))
            update[source_key] = value_to_load
            used_source_keys.add(source_key)

        for source_key, source_value in source.items():
            candidates = self._legacy_rpcm_key_candidates(source_key)[1:]
            if not candidates:
                continue
            loaded_targets = []
            for target_key in candidates:
                if target_key not in target or target_key in update:
                    continue
                if not hasattr(source_value, "shape") or source_value.shape != target[target_key].shape:
                    skipped_shape.append(
                        (
                            source_key,
                            target_key,
                            tuple(source_value.shape) if hasattr(source_value, "shape") else "<no-shape>",
                            tuple(target[target_key].shape),
                        )
                    )
                    continue
                value_to_load = source_value
                if (
                    source_key.startswith("roi_heads.box.predictor.cls_score.")
                    and is_detector_classifier_key(target_key)
                ):
                    value_to_load = reorder_detector_classifier_rows(
                        source_value,
                        source_order=BACKGROUND_LAST,
                        target_order=INTERNAL_DETECTOR_CLASS_ORDER,
                    )
                    reordered_detector_classifier.append((source_key, target_key))
                update[target_key] = value_to_load
                loaded_targets.append(target_key)
            if loaded_targets:
                used_source_keys.add(source_key)
                remapped[source_key] = loaded_targets

        loaded_predictor = [key for key in update if key.startswith(predictor_prefix)]
        if not loaded_predictor:
            raise RuntimeError(f"No compatible legacy RPCM predictor weights found in {path}")

        merged = dict(target)
        merged.update(update)
        incompatible = self.model.load_state_dict(merged, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Internal legacy RPCM initialization failed unexpectedly: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        ignored_unloaded_prefixes = (
            "roi_heads.relation.ppg.",
            "roi_heads.relation.ppn.",
        )
        unloaded_target = [
            key for key in target
            if key not in update and not key.startswith(ignored_unloaded_prefixes)
        ]
        unused_source = [
            key for key in source
            if key not in used_source_keys and not key.startswith("roi_heads.relation.PPG_HBB.")
        ]
        report = {
            "checkpoint": path,
            "loaded": len(update),
            "loaded_predictor": len(loaded_predictor),
            "remapped": len(remapped),
            "unloaded_target": len(unloaded_target),
            "unused_source": len(unused_source),
            "skipped_shape": len(skipped_shape),
            "reordered_detector_classifier": len(reordered_detector_classifier),
        }
        print("Legacy RPCM model-only initialization:", report, flush=True)
        if remapped:
            print("Legacy key remaps:", list(remapped.items())[:20], flush=True)
        if unloaded_target:
            print("Unloaded target keys sample:", unloaded_target[:30], flush=True)
        if unused_source:
            print("Unused source keys sample:", unused_source[:30], flush=True)
        if skipped_shape:
            print("Skipped shape mismatch sample:", skipped_shape[:20], flush=True)
        if reordered_detector_classifier:
            print(
                "Legacy detector classifier rows reordered (background_last->background_first):",
                reordered_detector_classifier,
                flush=True,
            )
        return {
            "checkpoint": path,
            "loaded": sorted(update),
            "loaded_predictor": sorted(loaded_predictor),
            "remapped": remapped,
            "unloaded_target": unloaded_target,
            "unused_source": unused_source,
            "skipped_shape": skipped_shape,
            "reordered_detector_classifier": reordered_detector_classifier,
        }

    def load_rpcm_predictor_weights(self, path: str):
        """Initialize the compatible TypedHyperRPCM blocks from an RPCM checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        source = self._checkpoint_state_dict(ckpt)
        target = self.model.state_dict()
        predictor_prefix = "roi_heads.relation.predictor."
        predictor_name = str(
            self.cfg.get("MODEL", {}).get("ROI_RELATION_HEAD", {}).get("PREDICTOR", "")
        )
        if predictor_name.upper() in {
            "RPCM_LEGACY",
            "LEGACY_RPCM",
            "RPCM_ORIGINAL_LEGACY",
            "ORIGINAL_RPCM_LEGACY",
            "RPCM_NATIVE_LEGACY",
        }:
            return self._load_legacy_rpcm_model_only(source, path)

        shared_prefixes = (
            "pairwise_feature_extractor.",
            "down_samp.",
            "rel_residual.",
            "rel_norm.",
            "proto_head.",
        )
        loaded, shape_mismatch, absent = [], [], []
        update = {}
        # Everything outside the predictor must migrate strictly. This includes
        # the detector, ROI/union extractors and relation post-processing state.
        ignored_external_prefixes = ("roi_heads.relation.ppg.", "roi_heads.relation.ppn.")
        external_missing = []
        for key, target_value in target.items():
            if key.startswith(predictor_prefix) or key.startswith(ignored_external_prefixes):
                continue
            if key not in source:
                external_missing.append(key)
            elif source[key].shape != target_value.shape:
                shape_mismatch.append((key, tuple(source[key].shape), tuple(target_value.shape)))
            else:
                update[key] = source[key]
                loaded.append(key)
        external_unexpected = [
            key for key in source
            if not key.startswith(predictor_prefix)
            and not key.startswith(ignored_external_prefixes)
            and key not in target
        ]
        if external_missing or external_unexpected or any(not key.startswith(predictor_prefix) for key, _, _ in shape_mismatch):
            raise RuntimeError(
                "RPCM checkpoint is incompatible outside the predictor: "
                f"missing={external_missing}, unexpected={external_unexpected}, "
                f"shape_mismatch={shape_mismatch}"
            )
        for key, value in source.items():
            if not key.startswith(predictor_prefix):
                continue
            local_key = key[len(predictor_prefix):]
            if not local_key.startswith(shared_prefixes):
                continue
            if key not in target:
                absent.append(key)
            elif target[key].shape != value.shape:
                shape_mismatch.append((key, tuple(value.shape), tuple(target[key].shape)))
            else:
                update[key] = value
                loaded.append(key)
        if not any(key.startswith(predictor_prefix) for key in loaded):
            raise RuntimeError(f"No compatible RPCM predictor weights found in {path}")
        target.update(update)
        self.model.load_state_dict(target, strict=True)
        report = {
            "checkpoint": path,
            "loaded": loaded,
            "shape_mismatch": shape_mismatch,
            "absent_in_target": absent,
        }
        print(
            "RPCM -> TypedHyperRPCM migration:",
            {"loaded": len(loaded), "shape_mismatch": len(shape_mismatch), "absent": len(absent)},
            flush=True,
        )
        for name in ("shape_mismatch", "absent_in_target"):
            if report[name]:
                print(f"  {name}: {report[name]}", flush=True)
        return report

    def load_typed_stage_weights(self, path: str):
        """Initialize from a TypedHyperRPCM checkpoint, allowing newly added aux heads."""
        ckpt = torch.load(path, map_location="cpu")
        source = self._checkpoint_state_dict(ckpt)
        target = self.model.state_dict()
        allowed_missing_prefixes = (
            "roi_heads.relation.ppg.",
            "roi_heads.relation.ppn.",
            "roi_heads.relation.predictor.vehicle_aux_head.",
            "roi_heads.relation.predictor.vehicle_label_embed.",
        )
        allowed_missing_keys = {
            "roi_heads.relation.predictor.vehicle_logit_scale",
            "roi_heads.relation.predictor.vehicle_pos_weight",
        }
        ignored_unexpected_prefixes = (
            "roi_heads.relation.ppg.",
            "roi_heads.relation.ppn.",
        )
        update, missing, shape_mismatch = {}, [], []
        for key, target_value in target.items():
            if key not in source:
                if key in allowed_missing_keys or key.startswith(allowed_missing_prefixes):
                    continue
                missing.append(key)
            elif source[key].shape != target_value.shape:
                shape_mismatch.append((key, tuple(source[key].shape), tuple(target_value.shape)))
            else:
                update[key] = source[key]
        unexpected = [
            key for key in source
            if key not in target and not key.startswith(ignored_unexpected_prefixes)
        ]
        if missing or shape_mismatch or unexpected:
            raise RuntimeError(
                "Typed checkpoint is incompatible: "
                f"missing={missing}, shape_mismatch={shape_mismatch}, unexpected={unexpected}"
            )
        target.update(update)
        self.model.load_state_dict(target, strict=True)
        print(
            "TypedHyperRPCM initialization:",
            {
                "checkpoint": path,
                "loaded": len(update),
                "new_aux_keys": len(target) - len(update),
            },
            flush=True,
        )
        return {"checkpoint": path, "loaded": list(update.keys())}

    def _move_targets(self, targets):
        return [t.to(self.device) for t in targets]

    def _sgdet_detector_inputs_from_metas(self, metas):
        """Build the raw-resolution detector stream for STAR Sgdets.

        The normal ``images`` batch is the relation-scale (typically 1024px)
        stream.  When a STAR sgdet dataset provides the original image/target
        pair in metadata, pad and move it independently for the frozen d1/d2
        patch detector.  Returning ``None`` preserves compatibility with
        external datasets and existing direct model callers.
        """
        if str(self.cfg["MODEL"].get("TASK", "")).lower() != "sgdet":
            return None, None
        raw_images = [meta.get("sgdet_detector_image") for meta in metas]
        raw_targets = [
            meta.get("sgdet_detector_target", meta.get("sgdet_detector_size"))
            for meta in metas
        ]
        if not raw_images or any(image is None for image in raw_images) or any(
            target is None for target in raw_targets
        ):
            return None, None
        # Keep full-resolution imagery on CPU.  SceneGraphDetector crops and
        # transfers only 1024px patches to CUDA, mirroring the original
        # detector and avoiding a giant padded (B,C,H,W) GPU allocation.
        return list(raw_images), list(raw_targets)

    def train(self, start_epoch: int = 0):
        val_split = str(self.cfg.get("SOLVER", {}).get("VAL_SPLIT", "val")).lower()
        loaders = self.dataloaders or build_dataloaders(
            self.cfg,
            splits=("train", val_split),
            shuffle_map={"train": True, val_split: False},
        )
        train_loader = loaders["train"]
        val_loader = loaders.get(val_split)
        if val_loader is None and val_split != "val":
            raise KeyError(
                f"SOLVER.VAL_SPLIT={val_split!r} was requested, but available loaders are {sorted(loaders.keys())}"
            )
        if val_loader is not None:
            print(f"Validation split: {val_split}", flush=True)
        epochs = self.cfg["SOLVER"]["MAX_EPOCHS"]
        accumulation_steps = max(
            1, int(self.cfg.get("SOLVER", {}).get("GRADIENT_ACCUMULATION_STEPS", 1))
        )
        optimizer_steps_per_epoch = max(1, (len(train_loader) + accumulation_steps - 1) // accumulation_steps)
        # Original SGG-Toolkit task scripts specify MAX_ITER / VAL_PERIOD in
        # optimizer steps.  Preserve that contract for the OBB compatibility
        # configs while retaining epoch-based behavior for existing runs.
        max_iterations = max(0, int(self.cfg.get("SOLVER", {}).get("MAX_ITER", 0)))
        iteration_compat = bool(self.cfg.get("SOLVER", {}).get("ITERATION_COMPAT", False))
        val_period_iter = max(
            0,
            int(
                self.cfg.get("SOLVER", {}).get(
                    "VAL_PERIOD" if iteration_compat else "VAL_PERIOD_ITER", 0
                )
            ),
        )
        val_start_iter = max(
            0,
            int(
                self.cfg.get("SOLVER", {}).get(
                    "VAL_START_PERIOD" if iteration_compat else "VAL_START_ITER", 0
                )
            ),
        )
        if val_period_iter > 0:
            self.val_period = max(1, math.ceil(val_period_iter / optimizer_steps_per_epoch))
        if val_start_iter > 0:
            self.val_start_period = max(1, math.ceil(val_start_iter / optimizer_steps_per_epoch))
        if max_iterations > 0:
            required_epochs = math.ceil(max(max_iterations - self.global_step, 0) / optimizer_steps_per_epoch)
            # MAX_ITER is the primary termination contract of the original
            # SGG-Toolkit scripts; MAX_EPOCHS remains an optional explicit
            # safety cap for smoke runs.
            epochs = min(int(epochs), int(start_epoch) + max(required_epochs, 1))
        best_recall_k = max(int(k) for k in self.cfg.get("TEST", {}).get("RECALL_AT", [100]))
        if self.warmup_iters <= 0 and self.warmup_epochs > 0:
            self.warmup_iters = int(round(self.warmup_epochs * optimizer_steps_per_epoch))
        if not self._scheduler_resolved and self.global_step == 0:
            self.scheduler = self._build_scheduler(num_iters_per_epoch=optimizer_steps_per_epoch)
        start_epoch = max(0, min(int(start_epoch), int(epochs)))
        epoch_durations: List[float] = []
        for epoch in range(start_epoch, epochs):
            if max_iterations > 0 and self.global_step >= max_iterations:
                break
            # PairGraphBuilder reads this value to apply the GT-injection schedule.
            self.cfg["_CURRENT_EPOCH"] = epoch + 1
            epoch_start = time.perf_counter()
            self.model.train()
            self._set_frozen_modules_eval()
            epoch_loss_sums: Dict[str, float] = {}
            epoch_loss_count = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", disable=_disable_tqdm_for_non_tty())
            self.optimizer.zero_grad(set_to_none=True)
            reached_max_iterations = False
            for batch_idx, (images, targets, metas) in enumerate(pbar):
                if max_iterations > 0 and self.global_step >= max_iterations:
                    reached_max_iterations = True
                    break
                images = images.to(self.device)
                targets = self._move_targets(targets)
                # Keep the raw detector stream separate from the resized
                # relation stream for STAR Sgdets; it is ignored elsewhere.
                # ``metas`` is intentionally retained by the data loader so
                # this does not alter the public batch tuple contract.
                detector_images, detector_targets = self._sgdet_detector_inputs_from_metas(metas)
                loss_dict = self.model(
                    images,
                    targets,
                    detector_images=detector_images,
                    detector_targets=detector_targets,
                )
                loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
                (loss / accumulation_steps).backward()
                grad_total_norm = None
                should_step = (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(train_loader)
                if should_step:
                    if self.grad_norm_clip > 0:
                        grad_total_norm = _clip_grad_norm(
                            self.model.named_parameters(),
                            max_norm=self.grad_norm_clip,
                            clip=True,
                            verbose=self.print_grad_freq > 0 and (self.global_step + 1) % self.print_grad_freq == 0,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if (
                        iteration_compat
                        and self.checkpoint_period > 0
                        and self.global_step % self.checkpoint_period == 0
                    ):
                        self.save_checkpoint(
                            epoch + 1,
                            f"model_iter_{self.global_step:07d}.pth",
                        )
                    if max_iterations > 0 and self.global_step >= max_iterations:
                        reached_max_iterations = True
                loss_values = {
                    k: float(v.detach().cpu())
                    for k, v in loss_dict.items()
                    if torch.is_tensor(v)
                }
                loss_values["loss_total"] = float(loss.detach().cpu())
                if grad_total_norm is not None:
                    loss_values["grad_norm"] = float(grad_total_norm)
                for key, value in loss_values.items():
                    epoch_loss_sums[key] = epoch_loss_sums.get(key, 0.0) + value
                epoch_loss_count += 1
                pbar.set_postfix(loss_values)
                if should_step and isinstance(
                    self.scheduler,
                    (WarmupMultiStepLR, WarmupCosineLR, WarmupLinearDecayLR, WarmupExponentialLR, NoOpLR),
                ):
                    self.scheduler.step()

            if epoch_loss_count > 0:
                avg_losses = {
                    key: value / epoch_loss_count
                    for key, value in sorted(epoch_loss_sums.items())
                }
                print(f"epoch {epoch + 1}/{epochs} train_loss:", avg_losses, flush=True)
                current_lr = max(float(group["lr"]) for group in self.optimizer.param_groups)
                print(f"epoch {epoch + 1}/{epochs} lr: {current_lr:.8g}", flush=True)

            train_elapsed = time.perf_counter() - epoch_start
            eval_elapsed = 0.0
            metrics = None
            if val_loader is not None and (epoch + 1) >= self.val_start_period and ((epoch + 1) % self.val_period == 0):
                eval_start = time.perf_counter()
                metrics = self.evaluate_loader(val_loader)
                eval_elapsed = time.perf_counter() - eval_start
                for metric_name in ("R", "mR", "HR"):
                    current_metric = float(metrics.get(metric_name, {}).get(best_recall_k, float("-inf")))
                    if current_metric > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = current_metric
                        self.save_checkpoint(epoch + 1, f"model_best_{metric_name}.pth", metrics=metrics)
                        print(
                            f"Saved best {metric_name} checkpoint: {metric_name}@{best_recall_k}={current_metric:.6f}",
                            flush=True,
                        )
            elif val_loader is not None:
                skip_reasons = []
                if (epoch + 1) < self.val_start_period:
                    skip_reasons.append(f"epoch {epoch + 1}/{epochs} < VAL_START_PERIOD={self.val_start_period}")
                if ((epoch + 1) % self.val_period) != 0:
                    skip_reasons.append(f"epoch {epoch + 1}/{epochs} not divisible by VAL_PERIOD={self.val_period}")
                reason_text = "; ".join(skip_reasons) if skip_reasons else "validation conditions not met"
                print(f"Validation skipped: {reason_text}", flush=True)
            if isinstance(self.scheduler, WarmupReduceLROnPlateau):
                plateau_metric = None if metrics is None else metrics.get("R", {}).get(best_recall_k)
                self.scheduler.step(plateau_metric, epoch=self.global_step)
            self.save_checkpoint(epoch + 1, "model_last.pth")
            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_elapsed)
            avg_epoch_elapsed = sum(epoch_durations) / max(len(epoch_durations), 1)
            eta_seconds = avg_epoch_elapsed * max(epochs - epoch - 1, 0)
            print(
                "Timing:",
                {
                    "epoch": f"{epoch + 1}/{epochs}",
                    "train": _format_duration(train_elapsed),
                    "eval": _format_duration(eval_elapsed),
                    "epoch_total": _format_duration(epoch_elapsed),
                    "eta": _format_duration(eta_seconds),
                },
                flush=True,
            )
            if reached_max_iterations:
                print(f"Reached SOLVER.MAX_ITER={max_iterations}; stopping training.", flush=True)
                break

    @torch.no_grad()
    def evaluate_loader(self, loader, return_result: bool = False):
        self.model.eval()
        preds_all = []
        gts_all = []
        metas_all = []
        graph_debug_enabled = bool(self.cfg.get("TEST", {}).get("GRAPH_DEBUG", {}).get("ENABLED", False))
        for batch_idx, (images, targets, metas) in enumerate(tqdm(loader, desc="Validation", disable=_disable_tqdm_for_non_tty())):
            images = images.to(self.device)
            targets = self._move_targets(targets)
            detector_images, detector_targets = self._sgdet_detector_inputs_from_metas(metas)
            batch_image_ids = []
            for target, meta in zip(targets, metas):
                image_id = meta.get("image_id")
                if image_id is None and target.has_field("image_id"):
                    image_id_tensor = target.get_field("image_id")
                    if image_id_tensor.numel() > 0:
                        image_id = int(image_id_tensor.reshape(-1)[0].item())
                batch_image_ids.append(int(image_id) if image_id is not None else -1)
            if graph_debug_enabled:
                print(
                    "EvalBatchDebug:",
                    {
                        "batch_idx": int(batch_idx),
                        "batch_size": int(len(targets)),
                        "image_ids": batch_image_ids,
                    },
                    flush=True,
                )
            try:
                preds = self.model(
                    images,
                    # Sgdet relation evaluation itself stays target-free, but
                    # SceneGraphDetector needs the resized targets here to
                    # convert raw detector boxes into relation-image coords.
                    targets,
                    detector_images=detector_images,
                    detector_targets=detector_targets,
                )
            except torch.cuda.OutOfMemoryError:
                if graph_debug_enabled:
                    print(
                        "EvalBatchDebug OOM:",
                        {
                            "batch_idx": int(batch_idx),
                            "batch_size": int(len(targets)),
                            "image_ids": batch_image_ids,
                        },
                        flush=True,
                    )
                raise
            preds_all.extend([p.to("cpu") for p in preds])
            gts_all.extend([t.to("cpu") for t in targets])
            metas_all.extend(metas)
        if any("tile_origin" in meta for meta in metas_all):
            preds_all, gts_all = _merge_tile_batches(
                preds_all,
                gts_all,
                metas_all,
                task=self.cfg["MODEL"]["TASK"],
                iou_thresh=float(self.cfg["TEST"].get("TILE_MERGE_IOU_THRESHOLD", self.cfg["TEST"].get("IOU_THRESHOLD", 0.5))),
            )
        iou_thresh = float(self.cfg["TEST"].get("IOU_THRESHOLD", 0.5))
        res = evaluate_sgg(
            preds_all,
            gts_all,
            mode=self.cfg["MODEL"]["TASK"],
            topk=tuple(self.cfg["TEST"]["RECALL_AT"]),
            iou_thresh=iou_thresh,
            enabled_metrics=("R", "mR", "A"),
        )
        harmonic_recall = {}
        for k, recall_value in res.recall.items():
            mean_recall_value = res.mean_recall.get(k, 0.0)
            denom = recall_value + mean_recall_value
            harmonic_recall[k] = 0.0 if denom <= 0 else float(2.0 * recall_value * mean_recall_value / denom)
        family_predicates = tuple(
            predicates
            for family, predicates in sorted(
                build_family_predicates(self.cfg["MODEL"]["ROI_RELATION_HEAD"]).items()
            )
            if family != 0
        )
        family_macro_recall = {}
        for k, per_predicate in res.per_predicate_recall.items():
            family_values = []
            for predicates in family_predicates:
                observed = [float(per_predicate[p]) for p in predicates if res.predicate_counts.get(p, 0) > 0]
                if observed:
                    family_values.append(sum(observed) / len(observed))
            family_macro_recall[k] = sum(family_values) / len(family_values) if family_values else 0.0
        metrics = {
            "R": res.recall,
            "mR": res.mean_recall,
            "HR": harmonic_recall,
            "family_macro_recall": family_macro_recall,
            "candidate-stage-coverage": res.candidate_stage_coverage,
            "predicate-candidate-stage-coverage": res.predicate_candidate_stage_coverage,
            "vehicle-aux": res.vehicle_aux_stats,
            "A": res.pair_accuracy if self.cfg["MODEL"]["TASK"] != "sgdet" else {},
            "images": res.num_images,
            "valid_images": res.valid_images,
        }
        predicate_names = getattr(getattr(loader, "dataset", None), "ind_to_predicates", None)
        for line in _build_eval_report_lines(
            metrics,
            res.per_predicate_recall,
            res.predicate_counts,
            predicate_names=predicate_names,
        ):
            print(line, flush=True)
        print(_format_metric_dict_row("family-mR", family_macro_recall, sorted(family_macro_recall)), flush=True)
        debug_cfg = self.cfg.get("TEST", {}).get("EVAL_DEBUG", {})
        if debug_cfg.get("ENABLED", False):
            best_k = max(int(k) for k in self.cfg["TEST"]["RECALL_AT"])
            predicate_rows = []
            for predicate, count in sorted(res.predicate_counts.items()):
                predicate_rows.append(
                    {
                        "predicate": int(predicate),
                        "count": int(count),
                        "recall": float(res.per_predicate_recall.get(best_k, {}).get(predicate, 0.0)),
                    }
                )
            coverage_rows = []
            focus_predicates = {
                int(v)
                for v in debug_cfg.get("CANDIDATE_PREDICATES", [])
                if int(v) > 0
            }
            for row in predicate_rows:
                predicate = int(row["predicate"])
                if focus_predicates and predicate not in focus_predicates:
                    continue
                coverage_rows.append(
                    {
                        **row,
                        "semantic": float(
                            res.predicate_candidate_stage_coverage.get("semantic", {}).get(predicate, 0.0)
                        ),
                        "ppn": float(
                            res.predicate_candidate_stage_coverage.get("ppn", {}).get(predicate, 0.0)
                        ),
                        "degree_cap": float(
                            res.predicate_candidate_stage_coverage.get("degree_cap", {}).get(predicate, 0.0)
                        ),
                        "final": float(
                            res.predicate_candidate_stage_coverage.get("final", {}).get(predicate, 0.0)
                        ),
                    }
                )
            predicate_rows.sort(
                key=lambda item: (
                    item["recall"],
                    -item["count"],
                    item["predicate"],
                )
            )
            coverage_rows.sort(
                key=lambda item: (
                    item["final"],
                    item["recall"],
                    -item["count"],
                    item["predicate"],
                )
            )
            confusion_rows = []
            for predicate, predicted_counts in sorted(res.predicate_pair_cls_confusion.items()):
                predicate = int(predicate)
                if focus_predicates and predicate not in focus_predicates:
                    continue
                top_predictions = sorted(
                    (
                        {"predicate": int(predicted), "count": int(count)}
                        for predicted, count in predicted_counts.items()
                    ),
                    key=lambda item: (-item["count"], item["predicate"]),
                )[: max(int(debug_cfg.get("CONFUSION_TOPK", 8)), 1)]
                confusion_rows.append(
                    {
                        "predicate": predicate,
                        "covered": int(sum(int(count) for count in predicted_counts.values())),
                        "predictions": top_predictions,
                    }
                )
            vehicle_aux_rows = []
            for predicate, stats in sorted(res.vehicle_aux_stats.items()):
                predicate = int(predicate)
                if focus_predicates and predicate not in focus_predicates:
                    continue
                vehicle_aux_rows.append({"predicate": predicate, **stats})
            image_rows = list(res.debug_rows)
            image_rows.sort(
                key=lambda item: (
                    float(item.get("triplet_recall", 0.0)),
                    -int(item.get("gt_rel_count", 0)),
                    int(item.get("image_id", -1)),
                )
            )
            debug_payload = {
                "best_k": best_k,
                "summary": {
                    "triplet_recall": float(res.recall.get(best_k, 0.0)),
                    "mean_recall": float(res.mean_recall.get(best_k, 0.0)),
                },
                "predicate_details": predicate_rows,
                "predicate_candidate_coverage": coverage_rows,
                "predicate_pair_cls_confusion": confusion_rows,
                "vehicle_aux": vehicle_aux_rows,
                "hardest_images": image_rows[: max(int(debug_cfg.get("TOP_IMAGES", 10)), 0)],
            }
            for line in _build_eval_debug_lines(
                {"debug": debug_payload},
                predicate_names=predicate_names,
                top_predicates=int(debug_cfg.get("TOP_PREDICATES", 10)),
                top_images=int(debug_cfg.get("TOP_IMAGES", 10)),
            ):
                print(line, flush=True)
        if return_result:
            return metrics, res
        return metrics
