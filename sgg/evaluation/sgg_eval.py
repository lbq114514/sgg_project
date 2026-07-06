from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


@dataclass
class SGGResult:
    recall: Dict[int, float]
    mean_recall: Dict[int, float]
    num_images: int
    valid_images: int
    per_predicate_recall: Dict[int, Dict[int, float]]
    predicate_counts: Dict[int, int]
    debug_rows: List[Dict[str, float | int]]
    no_graph_constraint_recall: Dict[int, float] = field(default_factory=dict)
    zero_shot_recall: Dict[int, float] = field(default_factory=dict)
    ng_zero_shot_recall: Dict[int, float] = field(default_factory=dict)
    pair_accuracy: Dict[int, float] = field(default_factory=dict)
    pair_recall: float = 0.0
    avg_pred_pairs: float = 0.0
    pair_precision: float = 0.0
    base_pair_recall: float = 0.0
    base_avg_pred_pairs: float = 0.0
    base_pair_precision: float = 0.0
    pruned_pair_recall: float = 0.0
    pruned_avg_pred_pairs: float = 0.0
    pruned_pair_precision: float = 0.0
    gt_pair_cls_recall: Dict[int, float] = field(default_factory=dict)
    gt_pair_cls_mean_recall: Dict[int, float] = field(default_factory=dict)
    gt_pair_coverage: float = 0.0
    predicate_gt_pair_cls_recall: Dict[int, Dict[int, float]] = field(default_factory=dict)
    predicate_gt_pair_cls_covered: Dict[int, int] = field(default_factory=dict)
    predicate_gt_pair_cls_accuracy: Dict[int, Dict[int, float]] = field(default_factory=dict)
    predicate_pair_cls_confusion: Dict[int, Dict[int, int]] = field(default_factory=dict)
    candidate_stage_coverage: Dict[str, float] = field(default_factory=dict)
    predicate_candidate_stage_coverage: Dict[str, Dict[int, float]] = field(default_factory=dict)
    vehicle_aux_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class _ImageContext:
    image_id: int
    gt_rels: torch.Tensor
    gt_labels: torch.Tensor
    gt_boxes: torch.Tensor
    pred_pair_idx: torch.Tensor
    rel_scores: torch.Tensor
    pred_obj_labels: torch.Tensor
    pred_obj_scores: torch.Tensor
    pred_boxes: torch.Tensor
    gt_pair_set: Set[Tuple[int, int]]


def _empty_long(shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.long)


def _empty_float(shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _get_gt_relations(target: BoxList) -> torch.Tensor:
    if target.has_field("relation_triplets"):
        return target.get_field("relation_triplets").long()
    if target.has_field("relation_tuple"):
        return target.get_field("relation_tuple").long()
    return _empty_long((0, 3))


def _get_pred_pair_idx(pred: BoxList) -> torch.Tensor:
    for field in ("rel_pair_idx", "rel_pair_idxs", "pair_idx"):
        if pred.has_field(field):
            return pred.get_field(field).long()
    return _empty_long((0, 2))


def _get_pred_rel_scores(pred: BoxList) -> torch.Tensor:
    if pred.has_field("pred_rel_scores"):
        return pred.get_field("pred_rel_scores").float()
    if pred.has_field("rel_scores"):
        return pred.get_field("rel_scores").float()
    if pred.has_field("rel_logits"):
        return pred.get_field("rel_logits").float().softmax(-1)
    return _empty_float((0, 0))


def _get_pred_object_state(pred: BoxList) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.has_field("pred_labels") and pred.has_field("pred_scores"):
        return pred.get_field("pred_labels").long(), pred.get_field("pred_scores").float()
    if pred.has_field("labels") and pred.has_field("scores"):
        return pred.get_field("labels").long(), pred.get_field("scores").float()
    if pred.has_field("pred_logits"):
        cls_scores = pred.get_field("pred_logits").float().softmax(-1)
        pred_scores, pred_labels = cls_scores[:, 1:].max(dim=-1)
        return pred_labels.long() + 1, pred_scores.float()
    if pred.has_field("labels"):
        labels = pred.get_field("labels").long()
        return labels, torch.ones_like(labels, dtype=torch.float32)
    return _empty_long((len(pred),)), _empty_float((len(pred),))


def _build_object_predictions(pred: BoxList, target: BoxList, mode: str) -> Tuple[torch.Tensor, torch.Tensor, BoxList]:
    gt_labels = target.get_field("labels").long()
    if mode == "predcls":
        return gt_labels, torch.ones_like(gt_labels, dtype=torch.float32), target

    pred_labels, pred_scores = _get_pred_object_state(pred)
    if mode == "sgcls":
        return pred_labels, pred_scores, target
    return pred_labels, pred_scores, pred


def _triplet(
    relations: torch.Tensor,
    classes: torch.Tensor,
    boxes: torch.Tensor,
    predicate_scores: Optional[torch.Tensor] = None,
    class_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if relations.numel() == 0:
        box_dim = boxes.size(-1) if boxes.ndim == 2 else 4
        return _empty_long((0, 3)), _empty_float((0, box_dim * 2)), None

    sub_id = relations[:, 0].long()
    obj_id = relations[:, 1].long()
    predicates = relations[:, 2].long()
    triplets = torch.stack([classes[sub_id], predicates, classes[obj_id]], dim=1)
    triplet_boxes = torch.cat([boxes[sub_id], boxes[obj_id]], dim=1)

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = torch.stack(
            [class_scores[sub_id], predicate_scores, class_scores[obj_id]],
            dim=1,
        )
    return triplets, triplet_boxes, triplet_scores


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return _empty_float((boxes1.size(0), boxes2.size(0)))

    box_dim = boxes1.size(-1)
    mode = "xyxy" if box_dim == 4 else "xywha"
    if box_dim == 5:
        try:
            boxlist1 = BoxList(boxes1, (1, 1), mode=mode)
            boxlist2 = BoxList(boxes2, (1, 1), mode=mode)
            return boxlist_iou(boxlist1, boxlist2, mode="auto")
        except Exception:
            half_w1 = boxes1[:, 2] * 0.5
            half_h1 = boxes1[:, 3] * 0.5
            half_w2 = boxes2[:, 2] * 0.5
            half_h2 = boxes2[:, 3] * 0.5
            boxes1 = torch.stack(
                [boxes1[:, 0] - half_w1, boxes1[:, 1] - half_h1, boxes1[:, 0] + half_w1, boxes1[:, 1] + half_h1],
                dim=1,
            )
            boxes2 = torch.stack(
                [boxes2[:, 0] - half_w2, boxes2[:, 1] - half_h2, boxes2[:, 0] + half_w2, boxes2[:, 1] + half_h2],
                dim=1,
            )
            mode = "xyxy"

    boxlist1 = BoxList(boxes1, (1, 1), mode=mode)
    boxlist2 = BoxList(boxes2, (1, 1), mode=mode)
    return boxlist_iou(boxlist1, boxlist2, mode="auto")


def _compute_pred_matches(
    gt_triplets: torch.Tensor,
    pred_triplets: torch.Tensor,
    gt_triplet_boxes: torch.Tensor,
    pred_triplet_boxes: torch.Tensor,
    iou_thresh: float,
    use_gt_boxes: bool = False,
    atol: float = 1e-4,
) -> List[List[int]]:
    if pred_triplets.numel() == 0:
        return []

    pred_to_gt: List[List[int]] = [[] for _ in range(pred_triplets.size(0))]
    if gt_triplets.numel() == 0:
        return pred_to_gt

    equality = (gt_triplets[:, None, :] == pred_triplets[None, :, :]).all(dim=-1)
    box_dim = gt_triplet_boxes.size(-1) // 2

    for gt_idx in range(gt_triplets.size(0)):
        pred_match = torch.nonzero(equality[gt_idx], as_tuple=False).flatten()
        if pred_match.numel() == 0:
            continue

        if use_gt_boxes:
            gt_box = gt_triplet_boxes[gt_idx : gt_idx + 1]
            boxes = pred_triplet_boxes[pred_match]
            keep = torch.isclose(boxes, gt_box, atol=atol, rtol=0.0).all(dim=1)
        else:
            gt_sub = gt_triplet_boxes[gt_idx : gt_idx + 1, :box_dim]
            gt_obj = gt_triplet_boxes[gt_idx : gt_idx + 1, box_dim:]
            pred_sub = pred_triplet_boxes[pred_match, :box_dim]
            pred_obj = pred_triplet_boxes[pred_match, box_dim:]

            sub_iou = _pairwise_iou(gt_sub, pred_sub).squeeze(0)
            obj_iou = _pairwise_iou(gt_obj, pred_obj).squeeze(0)
            keep = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
        for pred_idx in pred_match[keep].tolist():
            pred_to_gt[pred_idx].append(gt_idx)

    return pred_to_gt


def _relation_recall(pred_to_gt: Sequence[Sequence[int]], topk: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, Set[int]]]:
    matched_by_k: Dict[int, int] = {}
    matched_sets_by_k: Dict[int, Set[int]] = {}
    for k in topk:
        matched: Set[int] = set()
        for hits in pred_to_gt[:k]:
            matched.update(hits)
        matched_by_k[k] = len(matched)
        matched_sets_by_k[k] = matched
    return matched_by_k, matched_sets_by_k


def _build_image_context(pred: BoxList, target: BoxList, mode: str) -> Optional[_ImageContext]:
    gt_rels = _get_gt_relations(target)
    if gt_rels.numel() == 0:
        return None

    pred_obj_labels, pred_obj_scores, pred_boxlist = _build_object_predictions(pred, target, mode)
    num_objects = min(len(pred_obj_labels), len(pred_boxlist))
    pred_obj_labels = pred_obj_labels[:num_objects]
    pred_obj_scores = pred_obj_scores[:num_objects]
    pred_boxes = pred_boxlist.bbox.float()[:num_objects]

    pred_pair_idx = _get_pred_pair_idx(pred)
    rel_scores = _get_pred_rel_scores(pred)
    gt_pair_set = {(int(subj), int(obj)) for subj, obj, _ in gt_rels.tolist()}

    image_id = -1
    if target.has_field("image_id"):
        image_id_tensor = target.get_field("image_id")
        if image_id_tensor.numel() > 0:
            image_id = int(image_id_tensor.reshape(-1)[0].item())

    return _ImageContext(
        image_id=image_id,
        gt_rels=gt_rels,
        gt_labels=target.get_field("labels").long(),
        gt_boxes=target.bbox.float(),
        pred_pair_idx=pred_pair_idx,
        rel_scores=rel_scores,
        pred_obj_labels=pred_obj_labels,
        pred_obj_scores=pred_obj_scores,
        pred_boxes=pred_boxes,
        gt_pair_set=gt_pair_set,
    )


def _valid_pair_mask(pair_idx: torch.Tensor, num_objects: int) -> torch.Tensor:
    if pair_idx.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=pair_idx.device)
    return (
        (pair_idx[:, 0] >= 0)
        & (pair_idx[:, 0] < num_objects)
        & (pair_idx[:, 1] >= 0)
        & (pair_idx[:, 1] < num_objects)
        & (pair_idx[:, 0] != pair_idx[:, 1])
    )


def _graph_constrained_predictions(
    ctx: _ImageContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if ctx.pred_pair_idx.numel() == 0 or ctx.rel_scores.numel() == 0 or ctx.rel_scores.size(-1) <= 1:
        return _empty_long((0, 2)), _empty_long((0,)), _empty_float((0,)), _empty_float((0,))

    num_pairs = min(ctx.pred_pair_idx.size(0), ctx.rel_scores.size(0))
    pred_pair_idx = ctx.pred_pair_idx[:num_pairs]
    rel_scores = ctx.rel_scores[:num_pairs]

    pred_rels = rel_scores[:, 1:].argmax(dim=-1).long() + 1
    pred_rel_scores = rel_scores[:, 1:].max(dim=-1).values
    valid_mask = _valid_pair_mask(pred_pair_idx, len(ctx.pred_obj_labels))
    pred_pair_idx = pred_pair_idx[valid_mask]
    if pred_pair_idx.numel() == 0:
        return _empty_long((0, 2)), _empty_long((0,)), _empty_float((0,)), _empty_float((0,))
    pred_rels = pred_rels[valid_mask]
    pred_rel_scores = pred_rel_scores[valid_mask]
    triplet_scores = pred_rel_scores * ctx.pred_obj_scores[pred_pair_idx[:, 0]] * ctx.pred_obj_scores[pred_pair_idx[:, 1]]
    order = torch.argsort(triplet_scores, descending=True)
    return pred_pair_idx[order], pred_rels[order], pred_rel_scores[order], triplet_scores[order]


def _nogc_predictions(
    ctx: _ImageContext,
    max_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ctx.pred_pair_idx.numel() == 0 or ctx.rel_scores.numel() == 0 or ctx.rel_scores.size(-1) <= 1:
        return _empty_long((0, 2)), _empty_long((0,)), _empty_float((0,))

    num_pairs = min(ctx.pred_pair_idx.size(0), ctx.rel_scores.size(0))
    pred_pair_idx = ctx.pred_pair_idx[:num_pairs]
    rel_scores = ctx.rel_scores[:num_pairs]
    valid_mask = _valid_pair_mask(pred_pair_idx, len(ctx.pred_obj_labels))
    pred_pair_idx = pred_pair_idx[valid_mask]
    if pred_pair_idx.numel() == 0:
        return _empty_long((0, 2)), _empty_long((0,)), _empty_float((0,))
    rel_scores = rel_scores[valid_mask, 1:]
    obj_pair_scores = ctx.pred_obj_scores[pred_pair_idx[:, 0]] * ctx.pred_obj_scores[pred_pair_idx[:, 1]]
    overall_scores = obj_pair_scores[:, None] * rel_scores
    flat_scores = overall_scores.reshape(-1)
    if flat_scores.numel() == 0:
        return _empty_long((0, 2)), _empty_long((0,)), _empty_float((0,))

    top_count = min(max(int(max_rank), 1), int(flat_scores.numel()))
    top_scores, top_indices = torch.topk(flat_scores, k=top_count, sorted=True)
    pair_indices = top_indices // rel_scores.size(1)
    pred_rels = (top_indices % rel_scores.size(1)).long() + 1
    return pred_pair_idx[pair_indices], pred_rels, top_scores


def _match_predictions(
    ctx: _ImageContext,
    pred_pair_idx: torch.Tensor,
    pred_rels: torch.Tensor,
    pred_rel_scores: torch.Tensor,
    iou_thresh: float,
    use_gt_boxes: bool = False,
) -> List[List[int]]:
    if pred_pair_idx.numel() == 0:
        return []
    pred_relations = torch.cat([pred_pair_idx, pred_rels[:, None]], dim=1)
    gt_triplets, gt_triplet_boxes, _ = _triplet(ctx.gt_rels, ctx.gt_labels, ctx.gt_boxes)
    pred_triplets, pred_triplet_boxes, _ = _triplet(
        pred_relations,
        ctx.pred_obj_labels,
        ctx.pred_boxes,
        pred_rel_scores,
        ctx.pred_obj_scores,
    )
    return _compute_pred_matches(
        gt_triplets=gt_triplets,
        pred_triplets=pred_triplets,
        gt_triplet_boxes=gt_triplet_boxes,
        pred_triplet_boxes=pred_triplet_boxes,
        iou_thresh=iou_thresh,
        use_gt_boxes=use_gt_boxes,
    )


def _compute_zeroshot_indices(ctx: _ImageContext, zeroshot_triplets: Optional[torch.Tensor]) -> List[int]:
    if zeroshot_triplets is None or zeroshot_triplets.numel() == 0:
        return []
    gt_triplets = torch.stack(
        [
            ctx.gt_labels[ctx.gt_rels[:, 0].long()],
            ctx.gt_labels[ctx.gt_rels[:, 1].long()],
            ctx.gt_rels[:, 2].long(),
        ],
        dim=1,
    )
    equality = (gt_triplets[:, None, :] == zeroshot_triplets[None, :, :]).all(dim=-1)
    return torch.nonzero(equality.any(dim=1), as_tuple=False).flatten().tolist()


class _GraphRecallMetric:
    def __init__(self, topk: Sequence[int]):
        self.topk = tuple(topk)
        self.values = {k: [] for k in self.topk}

    def add(self, matched_by_k: Dict[int, int], num_gt_pairs: int) -> None:
        denom = max(int(num_gt_pairs), 1)
        for k in self.topk:
            self.values[k].append(float(matched_by_k.get(k, 0) / denom))

    def finalize(self) -> Dict[int, float]:
        return {k: _safe_mean(v) for k, v in self.values.items()}


class _MeanRecallMetric:
    def __init__(self, topk: Sequence[int], num_predicates: int):
        self.topk = tuple(topk)
        self.num_predicates = max(int(num_predicates), 0)
        self.predicate_recalls = {
            k: {predicate: [] for predicate in range(1, self.num_predicates + 1)}
            for k in self.topk
        }

    def add(self, gt_rels: torch.Tensor, matched_sets_by_k: Dict[int, Set[int]]) -> None:
        gt_rel_rows = gt_rels.tolist()

        for k in self.topk:
            matched = matched_sets_by_k.get(k, set())
            recall_hit = [0] * (self.num_predicates + 1)
            recall_count = [0] * (self.num_predicates + 1)

            for _, _, predicate in gt_rel_rows:
                predicate = int(predicate)
                if 0 <= predicate <= self.num_predicates:
                    recall_count[predicate] += 1

            for idx in matched:
                predicate = int(gt_rel_rows[idx][2])
                if 0 <= predicate <= self.num_predicates:
                    recall_hit[predicate] += 1

            for predicate in range(1, self.num_predicates + 1):
                if recall_count[predicate] > 0:
                    self.predicate_recalls[k][predicate].append(float(recall_hit[predicate] / recall_count[predicate]))

    def finalize(self) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
        mean_recall: Dict[int, float] = {}
        per_predicate_recall: Dict[int, Dict[int, float]] = {}
        for k in self.topk:
            per_predicate_recall[k] = {}
            recall_sum = 0.0
            for predicate in range(1, self.num_predicates + 1):
                predicate_recall = _safe_mean(self.predicate_recalls[k][predicate])
                per_predicate_recall[k][predicate] = predicate_recall
                recall_sum += predicate_recall
            mean_recall[k] = float(recall_sum / self.num_predicates) if self.num_predicates > 0 else 0.0
        return mean_recall, per_predicate_recall


class _ZeroShotRecallMetric:
    def __init__(self, topk: Sequence[int]):
        self.topk = tuple(topk)
        self.values = {k: [] for k in self.topk}

    def add(self, pred_to_gt: Sequence[Sequence[int]], zeroshot_idx: Sequence[int]) -> None:
        if not zeroshot_idx:
            return
        zeroshot_set = set(int(idx) for idx in zeroshot_idx)
        for k in self.topk:
            matched: Set[int] = set()
            for hits in pred_to_gt[:k]:
                matched.update(int(idx) for idx in hits)
            self.values[k].append(float(len(matched & zeroshot_set) / max(len(zeroshot_set), 1)))

    def finalize(self) -> Dict[int, float]:
        return {k: _safe_mean(v) for k, v in self.values.items()}


class _PairAccuracyMetric:
    def __init__(self, topk: Sequence[int]):
        self.topk = tuple(topk)
        self.values = {k: [] for k in self.topk}

    def add(
        self,
        mode: str,
        pred_to_gt: Sequence[Sequence[int]],
        sorted_pred_pairs: torch.Tensor,
        gt_pair_set: Set[Tuple[int, int]],
        gt_rel_count: int,
    ) -> None:
        if mode == "sgdet":
            return
        pair_mask = [
            (int(pair[0]), int(pair[1])) in gt_pair_set
            for pair in sorted_pred_pairs.tolist()
        ] if sorted_pred_pairs.numel() > 0 else []
        gt_pair_pred_to_gt = [hits for hits, keep in zip(pred_to_gt, pair_mask) if keep]
        for k in self.topk:
            matched: Set[int] = set()
            for hits in gt_pair_pred_to_gt[:k]:
                matched.update(int(idx) for idx in hits)
            self.values[k].append(float(len(matched) / max(int(gt_rel_count), 1)))

    def finalize(self) -> Dict[int, float]:
        return {k: _safe_mean(v) for k, v in self.values.items()}


def evaluate_sgg(
    predictions: List[BoxList],
    targets: List[BoxList],
    mode: str = "sgdet",
    topk: Sequence[int] = (20, 50, 100),
    iou_thresh: float = 0.5,
    enabled_metrics: Sequence[str] = ("R", "mR"),
    zeroshot_triplets: Optional[torch.Tensor] = None,
) -> SGGResult:
    if mode not in {"predcls", "sgcls", "sgdet"}:
        raise ValueError(f"Unsupported SGG evaluation mode: {mode}")

    topk = tuple(sorted(int(k) for k in topk))
    enabled = {str(name) for name in enabled_metrics}
    if zeroshot_triplets is not None:
        zeroshot_triplets = zeroshot_triplets.long().cpu()

    predicate_counts: Dict[int, int] = {}
    debug_rows: List[Dict[str, float | int]] = []
    valid_images = 0
    num_predicates = 0

    for pred in predictions:
        rel_scores = _get_pred_rel_scores(pred)
        if rel_scores.ndim == 2 and rel_scores.size(-1) > 1:
            num_predicates = max(num_predicates, int(rel_scores.size(-1) - 1))

    recall_metric = _GraphRecallMetric(topk)
    mean_recall_metric = _MeanRecallMetric(topk, num_predicates)
    nogc_metric = _GraphRecallMetric(topk) if "ngR" in enabled else None
    zeroshot_metric = _ZeroShotRecallMetric(topk) if "zR" in enabled else None
    ng_zeroshot_metric = _ZeroShotRecallMetric(topk) if "ng-zR" in enabled else None
    pair_accuracy_metric = _PairAccuracyMetric(topk) if "A" in enabled else None
    candidate_stage_hits = {"semantic": 0, "ppn": 0, "degree_cap": 0, "final": 0}
    predicate_candidate_hits = {stage: {} for stage in candidate_stage_hits}
    predicate_candidate_total: Dict[int, int] = {}
    predicate_pair_cls_confusion: Dict[int, Dict[int, int]] = {}
    vehicle_aux_total: Dict[int, int] = {}
    vehicle_aux_hits: Dict[int, int] = {}
    vehicle_aux_logit_sum: Dict[int, float] = {}
    candidate_stage_total = 0

    for pred, target in zip(predictions, targets):
        ctx = _build_image_context(pred, target, mode)
        if ctx is None:
            continue

        valid_images += 1
        gt_rows = ctx.gt_rels.tolist()
        candidate_stage_total += len(gt_rows)
        for row in gt_rows:
            predicate = int(row[2])
            predicate_candidate_total[predicate] = predicate_candidate_total.get(predicate, 0) + 1
        if pred.has_field("semantic_allowed_label_pairs"):
            allowed = pred.get_field("semantic_allowed_label_pairs").bool().reshape(
                pred.get_field("semantic_allowed_label_pairs").shape[-2:]
            )
            for row in gt_rows:
                hit = bool(allowed[int(ctx.gt_labels[int(row[0])]), int(ctx.gt_labels[int(row[1])])])
                candidate_stage_hits["semantic"] += int(hit)
                if hit:
                    predicate = int(row[2])
                    predicate_candidate_hits["semantic"][predicate] = predicate_candidate_hits["semantic"].get(predicate, 0) + 1
        stage_fields = {
            "ppn": "ppn_ranked_pair_idxs",
            "degree_cap": "degree_capped_pair_idxs",
        }
        for stage, field_name in stage_fields.items():
            if pred.has_field(field_name):
                stage_pairs = {tuple(pair) for pair in pred.get_field(field_name).long().tolist()}
                for row in gt_rows:
                    hit = (int(row[0]), int(row[1])) in stage_pairs
                    candidate_stage_hits[stage] += int(hit)
                    if hit:
                        predicate = int(row[2])
                        predicate_candidate_hits[stage][predicate] = predicate_candidate_hits[stage].get(predicate, 0) + 1
        final_pairs = {tuple(pair) for pair in ctx.pred_pair_idx.tolist()}
        for row in gt_rows:
            hit = (int(row[0]), int(row[1])) in final_pairs
            candidate_stage_hits["final"] += int(hit)
            if hit:
                predicate = int(row[2])
                predicate_candidate_hits["final"][predicate] = predicate_candidate_hits["final"].get(predicate, 0) + 1
        for _, _, predicate in ctx.gt_rels.tolist():
            predicate = int(predicate)
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1

        sorted_pairs, sorted_rels, sorted_rel_scores, _ = _graph_constrained_predictions(ctx)
        if pred.has_field("vehicle_aux_logits") and pred.has_field("vehicle_aux_predicates"):
            aux_logits = pred.get_field("vehicle_aux_logits").float()
            aux_predicates = pred.get_field("vehicle_aux_predicates").long().tolist()
            aux_cols = {int(predicate): col for col, predicate in enumerate(aux_predicates)}
            raw_rels = (
                target.get_field("all_relation_triplets").long()
                if target.has_field("all_relation_triplets")
                else ctx.gt_rels
            )
            pair_to_row = {
                (int(pair[0]), int(pair[1])): row
                for row, pair in enumerate(ctx.pred_pair_idx.tolist()[: aux_logits.size(0)])
            }
            for head, tail, predicate in raw_rels.tolist():
                col = aux_cols.get(int(predicate))
                row = pair_to_row.get((int(head), int(tail)))
                if col is None or row is None:
                    continue
                value = float(aux_logits[row, col].item())
                vehicle_aux_total[int(predicate)] = vehicle_aux_total.get(int(predicate), 0) + 1
                vehicle_aux_hits[int(predicate)] = vehicle_aux_hits.get(int(predicate), 0) + int(value > 0.0)
                vehicle_aux_logit_sum[int(predicate)] = vehicle_aux_logit_sum.get(int(predicate), 0.0) + value
        pair_to_predicate: Dict[Tuple[int, int], int] = {}
        pair_to_score: Dict[Tuple[int, int], float] = {}
        for pair, predicate, score in zip(
            sorted_pairs.tolist(),
            sorted_rels.tolist(),
            sorted_rel_scores.tolist(),
        ):
            key = (int(pair[0]), int(pair[1]))
            if key not in pair_to_predicate or float(score) > pair_to_score[key]:
                pair_to_predicate[key] = int(predicate)
                pair_to_score[key] = float(score)
        for head, tail, predicate in gt_rows:
            key = (int(head), int(tail))
            if key in pair_to_predicate:
                gt_predicate = int(predicate)
                pred_predicate = int(pair_to_predicate[key])
                bucket = predicate_pair_cls_confusion.setdefault(gt_predicate, {})
                bucket[pred_predicate] = bucket.get(pred_predicate, 0) + 1

        use_gt_boxes = mode in {"predcls", "sgcls"}
        pred_to_gt = _match_predictions(
            ctx,
            sorted_pairs,
            sorted_rels,
            sorted_rel_scores,
            iou_thresh,
            use_gt_boxes=use_gt_boxes,
        )
        matched_by_k, matched_sets_by_k = _relation_recall(pred_to_gt, topk)

        if "R" in enabled:
            recall_metric.add(matched_by_k, len(ctx.gt_pair_set))
        if "mR" in enabled:
            mean_recall_metric.add(ctx.gt_rels, matched_sets_by_k)
        if pair_accuracy_metric is not None:
            pair_accuracy_metric.add(mode, pred_to_gt, sorted_pairs, ctx.gt_pair_set, int(ctx.gt_rels.size(0)))

        zeroshot_idx = _compute_zeroshot_indices(ctx, zeroshot_triplets)
        if zeroshot_metric is not None:
            zeroshot_metric.add(pred_to_gt, zeroshot_idx)

        if nogc_metric is not None or ng_zeroshot_metric is not None:
            ng_pairs, ng_rels, ng_scores = _nogc_predictions(ctx, max(max(topk), 2000))
            ng_pred_to_gt = _match_predictions(
                ctx,
                ng_pairs,
                ng_rels,
                ng_scores,
                iou_thresh,
                use_gt_boxes=use_gt_boxes,
            )
            ng_matched_by_k, _ = _relation_recall(ng_pred_to_gt, topk)
            if nogc_metric is not None:
                nogc_metric.add(ng_matched_by_k, len(ctx.gt_pair_set))
            if ng_zeroshot_metric is not None:
                ng_zeroshot_metric.add(ng_pred_to_gt, zeroshot_idx)

        best_k = topk[-1]
        debug_rows.append(
            {
                "image_id": int(ctx.image_id),
                "gt_rel_count": int(ctx.gt_rels.size(0)),
                "gt_pair_count": int(len(ctx.gt_pair_set)),
                "pred_pair_count": int(ctx.pred_pair_idx.size(0)),
                "triplet_match_count": int(matched_by_k.get(best_k, 0)),
                "triplet_recall": float(matched_by_k.get(best_k, 0) / max(len(ctx.gt_pair_set), 1)),
            }
        )

    recall = recall_metric.finalize() if "R" in enabled else {k: 0.0 for k in topk}
    if "mR" in enabled:
        mean_recall, per_predicate_recall = mean_recall_metric.finalize()
    else:
        mean_recall = {k: 0.0 for k in topk}
        per_predicate_recall = {k: {} for k in topk}

    return SGGResult(
        recall=recall,
        mean_recall=mean_recall,
        num_images=len(predictions),
        valid_images=valid_images,
        per_predicate_recall=per_predicate_recall,
        predicate_counts={int(k): int(v) for k, v in predicate_counts.items()},
        debug_rows=debug_rows,
        no_graph_constraint_recall=nogc_metric.finalize() if nogc_metric is not None else {},
        zero_shot_recall=zeroshot_metric.finalize() if zeroshot_metric is not None else {},
        ng_zero_shot_recall=ng_zeroshot_metric.finalize() if ng_zeroshot_metric is not None else {},
        pair_accuracy=pair_accuracy_metric.finalize() if pair_accuracy_metric is not None else {},
        candidate_stage_coverage={
            stage: float(hits / max(candidate_stage_total, 1))
            for stage, hits in candidate_stage_hits.items()
        },
        predicate_candidate_stage_coverage={
            stage: {
                int(predicate): float(hits.get(predicate, 0) / max(total, 1))
                for predicate, total in predicate_candidate_total.items()
            }
            for stage, hits in predicate_candidate_hits.items()
        },
        predicate_pair_cls_confusion={
            int(predicate): {int(predicted): int(count) for predicted, count in predicted_counts.items()}
            for predicate, predicted_counts in predicate_pair_cls_confusion.items()
        },
        vehicle_aux_stats={
            int(predicate): {
                "pos": float(total),
                "aux_recall_at_0": float(vehicle_aux_hits.get(predicate, 0) / max(total, 1)),
                "avg_logit": float(vehicle_aux_logit_sum.get(predicate, 0.0) / max(total, 1)),
            }
            for predicate, total in vehicle_aux_total.items()
        },
    )
