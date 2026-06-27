from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


def rbbox2roi(boxes_list: Sequence[BoxList]) -> torch.Tensor:
    rois = []
    for img_id, boxlist in enumerate(boxes_list):
        if len(boxlist) == 0:
            continue
        img_inds = boxlist.bbox.new_full((len(boxlist), 1), img_id)
        rois.append(torch.cat([img_inds, boxlist.bbox], dim=1))
    if not rois:
        device = boxes_list[0].bbox.device if boxes_list else torch.device("cpu")
        return torch.zeros((0, 6), dtype=torch.float32, device=device)
    return torch.cat(rois, dim=0)


class OrientedStandardRoIHead(nn.Module):
    """
    Lightweight oriented RoI head with one bbox branch.

    The implementation intentionally mirrors the useful behavior of mmrotate's
    standard RoI head while using this project's BoxList and feature-extractor
    abstractions.
    """

    def __init__(
        self,
        bbox_roi_extractor: nn.Module,
        bbox_head: nn.Module,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.5,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
    ):
        super().__init__()
        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head
        self.fg_iou_thresh = float(fg_iou_thresh)
        self.bg_iou_thresh = float(bg_iou_thresh)
        self.batch_size_per_image = int(batch_size_per_image)
        self.positive_fraction = float(positive_fraction)

    def forward_train(
        self,
        x,
        proposals: Sequence[BoxList],
        targets: Sequence[BoxList],
    ) -> Dict[str, torch.Tensor]:
        sampled_boxes, labels, label_weights, bbox_targets, bbox_weights = self._build_targets(
            proposals,
            targets,
        )
        rois = rbbox2roi(sampled_boxes)
        bbox_results = self._bbox_forward(x, sampled_boxes, rois)
        loss_bbox = self.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
        )
        bbox_results["loss_bbox"] = loss_bbox
        bbox_results["rois"] = rois
        bbox_results["sampled_boxes"] = sampled_boxes
        return bbox_results

    def simple_test_bboxes(
        self,
        x,
        proposals: Sequence[BoxList],
        rcnn_test_cfg: Dict | None = None,
        rescale: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, proposals, rois)
        cls_score = bbox_results["cls_score"]
        bbox_pred = bbox_results["bbox_pred"]

        num_proposals_per_img = [len(p) for p in proposals]
        rois_split = rois.split(num_proposals_per_img, dim=0)
        cls_split = cls_score.split(num_proposals_per_img, dim=0)
        if bbox_pred is None:
            bbox_split = [None] * len(proposals)
        else:
            bbox_split = bbox_pred.split(num_proposals_per_img, dim=0)

        det_bboxes = []
        det_labels = []
        for per_img_rois, per_img_cls, per_img_bbox, proposal in zip(
            rois_split,
            cls_split,
            bbox_split,
            proposals,
        ):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                per_img_rois,
                per_img_cls,
                per_img_bbox,
                img_shape=(proposal.size[1], proposal.size[0]),
                rescale=rescale,
                cfg=rcnn_test_cfg,
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _bbox_forward(self, x, proposals: Sequence[BoxList], rois: torch.Tensor):
        bbox_feats = self.bbox_roi_extractor(x, list(proposals))
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        return {
            "bbox_feats": bbox_feats,
            "cls_score": cls_score,
            "bbox_pred": bbox_pred,
            "rois": rois,
        }

    def _build_targets(
        self,
        proposals: Sequence[BoxList],
        targets: Sequence[BoxList],
    ):
        sampled_boxes = []
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []

        for proposal, target in zip(proposals, targets):
            (
                sampled_boxlist,
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
            ) = self._build_targets_single(proposal, target)
            sampled_boxes.append(sampled_boxlist)
            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)

        if labels_list:
            labels = torch.cat(labels_list, dim=0)
            label_weights = torch.cat(label_weights_list, dim=0)
            bbox_targets = torch.cat(bbox_targets_list, dim=0)
            bbox_weights = torch.cat(bbox_weights_list, dim=0)
        else:
            device = proposals[0].bbox.device if proposals else torch.device("cpu")
            labels = torch.zeros((0,), dtype=torch.long, device=device)
            label_weights = torch.zeros((0,), dtype=torch.float32, device=device)
            bbox_targets = torch.zeros((0, 5), dtype=torch.float32, device=device)
            bbox_weights = torch.zeros((0, 5), dtype=torch.float32, device=device)

        return sampled_boxes, labels, label_weights, bbox_targets, bbox_weights

    def _build_targets_single(
        self,
        proposal: BoxList,
        target: BoxList,
    ):
        device = proposal.bbox.device
        num_props = len(proposal)
        if num_props == 0:
            empty = proposal.bbox.new_zeros((0, 5))
            sampled = BoxList(empty, proposal.size, proposal.mode)
            return (
                sampled,
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
                torch.zeros((0, 5), dtype=torch.float32, device=device),
                torch.zeros((0, 5), dtype=torch.float32, device=device),
            )

        if len(target) == 0:
            neg_inds = self._sample_indices(
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.arange(num_props, device=device),
                device,
            )
            sampled_boxlist = proposal[neg_inds]
            labels = torch.zeros((len(neg_inds),), dtype=torch.long, device=device)
            label_weights = torch.ones((len(neg_inds),), dtype=torch.float32, device=device)
            bbox_targets = torch.zeros((len(neg_inds), 5), dtype=torch.float32, device=device)
            bbox_weights = torch.zeros((len(neg_inds), 5), dtype=torch.float32, device=device)
            return sampled_boxlist, labels, label_weights, bbox_targets, bbox_weights

        ious = boxlist_iou(proposal, target, mode="obb")
        matched_ious, matched_inds = ious.max(dim=1)

        pos_inds = torch.nonzero(matched_ious >= self.fg_iou_thresh, as_tuple=False).squeeze(1)
        neg_inds = torch.nonzero(matched_ious < self.bg_iou_thresh, as_tuple=False).squeeze(1)
        sampled_inds = self._sample_indices(pos_inds, neg_inds, device)
        sampled_boxlist = proposal[sampled_inds]

        sampled_pos_mask = matched_ious[sampled_inds] >= self.fg_iou_thresh
        sampled_matched_inds = matched_inds[sampled_inds]
        matched_gt = target[sampled_matched_inds]
        matched_labels = target.get_field("labels")[sampled_matched_inds].long().to(device)

        labels = torch.zeros((len(sampled_inds),), dtype=torch.long, device=device)
        labels[sampled_pos_mask] = matched_labels[sampled_pos_mask]
        label_weights = torch.ones((len(sampled_inds),), dtype=torch.float32, device=device)

        bbox_targets = torch.zeros((len(sampled_inds), 5), dtype=torch.float32, device=device)
        bbox_weights = torch.zeros((len(sampled_inds), 5), dtype=torch.float32, device=device)
        if sampled_pos_mask.any():
            bbox_targets[sampled_pos_mask] = self.bbox_head.bbox_coder.encode(
                sampled_boxlist.bbox[sampled_pos_mask],
                matched_gt.bbox[sampled_pos_mask],
            )
            bbox_weights[sampled_pos_mask] = 1.0

        return sampled_boxlist, labels, label_weights, bbox_targets, bbox_weights

    def _sample_indices(self, pos_inds: torch.Tensor, neg_inds: torch.Tensor, device) -> torch.Tensor:
        num_pos = min(int(self.batch_size_per_image * self.positive_fraction), pos_inds.numel())
        num_neg = min(self.batch_size_per_image - num_pos, neg_inds.numel())

        if pos_inds.numel() > 0:
            pos_perm = torch.randperm(pos_inds.numel(), device=device)[:num_pos]
            pos_inds = pos_inds[pos_perm]
        if neg_inds.numel() > 0:
            neg_perm = torch.randperm(neg_inds.numel(), device=device)[:num_neg]
            neg_inds = neg_inds[neg_perm]

        if pos_inds.numel() == 0:
            return neg_inds
        if neg_inds.numel() == 0:
            return pos_inds
        return torch.cat([pos_inds, neg_inds], dim=0)
