from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from sgg.modeling.core.obb_ops import get_boxlist_angle_unit, set_boxlist_angle_unit
from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


def obj_prediction_nms(
    proposal: BoxList,
    boxes_per_cls: torch.Tensor,
    pred_logits: torch.Tensor,
    nms_thresh: float = 0.3,
) -> torch.Tensor:
    """Memory-bounded equivalent of the original RPCM late object NMS.

    The source implementation materializes ``N x N x C`` overlaps.  This
    implementation preserves its global score ordering and per-class
    suppression without constructing that tensor, which matters for dense
    STAR sgdet proposals.
    """
    num_obj, num_cls = pred_logits.shape
    if num_obj == 0:
        return pred_logits.new_zeros((0,), dtype=torch.long)
    if boxes_per_cls.shape[:2] != (num_obj, num_cls):
        raise ValueError(
            "boxes_per_cls must have shape [num_obj, num_cls, box_dim], "
            f"got {tuple(boxes_per_cls.shape)} for logits {tuple(pred_logits.shape)}"
        )

    probabilities = F.softmax(pred_logits, dim=1).clone()
    probabilities[:, 0] = 0.0
    order = torch.argsort(probabilities.reshape(-1), descending=True).cpu().tolist()
    # The greedy bookkeeping is tiny and intentionally kept on CPU.  Python
    # indexing into CUDA booleans would otherwise synchronize once per
    # candidate and make late NMS unusably slow.
    assigned = torch.zeros((num_obj,), dtype=torch.bool)
    suppressed = torch.zeros((num_obj, num_cls), dtype=torch.bool)
    pred_labels = torch.zeros((num_obj,), dtype=torch.long)
    box_mode = proposal.mode
    angle_unit = get_boxlist_angle_unit(proposal, "degree")

    # This STAR detector is class-agnostic for box regression, so every class
    # channel normally contains the same box.  Compute the N x N overlap once
    # in bounded row chunks and keep only a boolean CPU matrix, instead of the
    # source implementation's N x N x C float tensor.
    class_agnostic = torch.equal(
        boxes_per_cls,
        boxes_per_cls[:, :1, :].expand_as(boxes_per_cls),
    )
    shared_overlaps = None
    if class_agnostic:
        candidates = BoxList(boxes_per_cls[:, 0], proposal.size, box_mode)
        if box_mode == "xywha":
            set_boxlist_angle_unit(candidates, angle_unit)
        overlap_rows = []
        for start in range(0, num_obj, 256):
            rows = BoxList(
                boxes_per_cls[start : start + 256, 0], proposal.size, box_mode
            )
            if box_mode == "xywha":
                set_boxlist_angle_unit(rows, angle_unit)
            overlap_rows.append(
                (boxlist_iou(rows, candidates, mode="auto") >= float(nms_thresh))
                .cpu()
            )
        shared_overlaps = torch.cat(overlap_rows, dim=0)
        if box_mode == "xywha":
            # Preserve SGG-Toolkit's OBB implementation exactly: its
            # nms_overlaps_rotated fills only i<j entries.  Although unusual,
            # this ordering is part of the reported sgdet behavior.
            indices = torch.arange(num_obj)
            shared_overlaps &= indices[None, :] > indices[:, None]

    for flat_index in order:
        box_index = int(flat_index // num_cls)
        class_index = int(flat_index % num_cls)
        if class_index == 0 or assigned[box_index] or suppressed[box_index, class_index]:
            continue
        pred_labels[box_index] = class_index
        assigned[box_index] = True

        if shared_overlaps is not None:
            overlap_mask = shared_overlaps[box_index]
        else:
            chosen = BoxList(
                boxes_per_cls[box_index : box_index + 1, class_index],
                proposal.size,
                box_mode,
            )
            candidates = BoxList(
                boxes_per_cls[:, class_index], proposal.size, box_mode
            )
            if box_mode == "xywha":
                set_boxlist_angle_unit(chosen, angle_unit)
                set_boxlist_angle_unit(candidates, angle_unit)
            overlap_mask = (
                boxlist_iou(chosen, candidates, mode="auto").squeeze(0)
                >= float(nms_thresh)
            ).cpu()
            if box_mode == "xywha":
                overlap_mask &= torch.arange(num_obj) > box_index
        suppressed[:, class_index] |= overlap_mask
        if bool(assigned.all()):
            break
    return pred_labels.to(pred_logits.device)


class RelationPostProcessor:
    def __init__(self, cfg: dict):
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.use_gt_box = bool(rel_cfg.get("USE_GT_BOX", False))
        self.pairness_score_weight = float(rel_cfg.get("HIER_PAIRNESS_SCORE_WEIGHT", 0.0))
        self.later_nms_pred_thres = float(
            rel_cfg.get(
                "LATER_NMS_PREDICTION_THRES",
                cfg.get("TEST", {})
                .get("RELATION", {})
                .get("LATER_NMS_PREDICTION_THRES", 0.3),
            )
        )

    def _split_relation_logits(self, relation_logits, rel_pair_idxs):
        if isinstance(relation_logits, torch.Tensor):
            rel_splits = [pairs.size(0) for pairs in rel_pair_idxs]
            return list(relation_logits.split(rel_splits, dim=0))
        return relation_logits

    def _split_refine_logits(self, refine_logits, proposals: Sequence):
        if refine_logits is None:
            return [None] * len(proposals)
        if isinstance(refine_logits, torch.Tensor):
            obj_splits = [len(proposal) for proposal in proposals]
            return list(refine_logits.split(obj_splits, dim=0))
        return list(refine_logits)

    def _object_predictions(
        self,
        proposal: BoxList,
        obj_logit: torch.Tensor | None,
    ) -> tuple[BoxList, torch.Tensor, torch.Tensor]:
        if obj_logit is None:
            if proposal.has_field("predict_logits"):
                obj_logit = proposal.get_field("predict_logits")
            elif proposal.has_field("pred_logits"):
                obj_logit = proposal.get_field("pred_logits")

        if obj_logit is None or obj_logit.numel() == 0:
            pred_labels = proposal.get_field("pred_labels") if proposal.has_field("pred_labels") else proposal.bbox.new_zeros((len(proposal),), dtype=torch.long)
            pred_scores = proposal.get_field("pred_scores") if proposal.has_field("pred_scores") else proposal.bbox.new_zeros((len(proposal),))
            return proposal, pred_labels.long(), pred_scores.float()

        obj_class_prob = F.softmax(obj_logit, dim=-1)
        if obj_class_prob.size(1) > 0:
            obj_class_prob = obj_class_prob.clone()
            obj_class_prob[:, 0] = 0.0

        if not self.use_gt_box and proposal.has_field("boxes_per_cls") and len(proposal) > 0:
            obj_pred = obj_prediction_nms(
                proposal,
                proposal.get_field("boxes_per_cls"),
                obj_logit,
                self.later_nms_pred_thres,
            )
            obj_scores = obj_class_prob.gather(1, obj_pred[:, None]).squeeze(1)
        else:
            obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
            obj_pred = obj_pred + 1
        boxlist = proposal

        if not self.use_gt_box and proposal.has_field("boxes_per_cls") and len(proposal) > 0:
            boxes_per_cls = proposal.get_field("boxes_per_cls")
            class_inds = obj_pred.clamp(min=0, max=boxes_per_cls.size(1) - 1)
            selected_boxes = boxes_per_cls[torch.arange(len(proposal), device=obj_pred.device), class_inds]
            boxlist = BoxList(selected_boxes, proposal.size, proposal.mode)
            for field in proposal.fields():
                if field == "boxes_per_cls":
                    continue
                value = proposal.get_field(field)
                boxlist.add_field(field, value)

        boxlist.add_field("pred_labels", obj_pred.long())
        boxlist.add_field("pred_scores", obj_scores.float())
        boxlist.add_field("scores", obj_scores.float())
        boxlist.add_field("predict_logits", obj_logit)
        boxlist.add_field("pred_logits", obj_logit)
        return boxlist, obj_pred.long(), obj_scores.float()

    def __call__(self, x, rel_pair_idxs, proposals: Sequence):
        if isinstance(x, tuple):
            relation_logits, refine_logits = x
        else:
            relation_logits, refine_logits = x, None

        relation_logits = self._split_relation_logits(relation_logits, rel_pair_idxs)
        refine_logits = self._split_refine_logits(refine_logits, proposals)

        results = []
        for proposal, pair_idx, rel_logit, obj_logit in zip(proposals, rel_pair_idxs, relation_logits, refine_logits):
            boxlist, obj_pred, obj_scores = self._object_predictions(proposal, obj_logit)
            boxlist.add_field("rel_pair_idxs", pair_idx)

            if rel_logit is None or rel_logit.numel() == 0 or pair_idx.numel() == 0:
                boxlist.add_field("pred_rel_scores", boxlist.bbox.new_zeros((0, 1)))
                boxlist.add_field("pred_rel_labels", torch.zeros((0,), dtype=torch.long, device=boxlist.bbox.device))
                if boxlist.has_field("vehicle_aux_logits"):
                    boxlist.add_field("vehicle_aux_logits", boxlist.get_field("vehicle_aux_logits")[:0])
                results.append(boxlist)
                continue

            rel_scores = torch.softmax(rel_logit, dim=1)
            if (
                self.pairness_score_weight > 0
                and rel_scores.size(1) > 1
                and boxlist.has_field("pred_pairness_scores")
            ):
                pairness_scores = boxlist.get_field("pred_pairness_scores").to(
                    device=rel_scores.device,
                    dtype=rel_scores.dtype,
                )
                if pairness_scores.numel() >= rel_scores.size(0):
                    pairness_scores = pairness_scores[: rel_scores.size(0)].clamp(min=1e-6, max=1.0)
                    rel_scores = rel_scores.clone()
                    rel_scores[:, 1:] = rel_scores[:, 1:] * pairness_scores.pow(self.pairness_score_weight).unsqueeze(1)

            rel_conf, rel_label = rel_scores[:, 1:].max(dim=1)
            rel_label = rel_label + 1
            triple_scores = rel_conf * obj_scores[pair_idx[:, 0]] * obj_scores[pair_idx[:, 1]]
            sorting_idx = torch.argsort(triple_scores, descending=True)

            boxlist.add_field("rel_pair_idxs", pair_idx[sorting_idx])
            boxlist.add_field("pred_rel_scores", rel_scores[sorting_idx])
            boxlist.add_field("pred_rel_labels", rel_label[sorting_idx])
            if boxlist.has_field("vehicle_aux_logits"):
                vehicle_aux_logits = boxlist.get_field("vehicle_aux_logits")
                if vehicle_aux_logits.numel() > 0 and vehicle_aux_logits.size(0) >= sorting_idx.numel():
                    boxlist.add_field("vehicle_aux_logits", vehicle_aux_logits[: sorting_idx.numel()][sorting_idx])
            results.append(boxlist)
        return results


def make_roi_relation_post_processor(cfg: dict):
    return RelationPostProcessor(cfg)
