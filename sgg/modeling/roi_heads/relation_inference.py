from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from sgg.structures.boxes import BoxList


class RelationPostProcessor:
    def __init__(self, cfg: dict):
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.use_gt_box = bool(rel_cfg.get("USE_GT_BOX", False))
        self.pairness_score_weight = float(rel_cfg.get("HIER_PAIRNESS_SCORE_WEIGHT", 0.0))

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
            results.append(boxlist)
        return results


def make_roi_relation_post_processor(cfg: dict):
    return RelationPostProcessor(cfg)
