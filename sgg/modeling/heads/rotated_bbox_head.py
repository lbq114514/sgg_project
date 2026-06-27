from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.modeling.box_coder.box_coder import DeltaXYWHAOBBoxCoder
from sgg.modeling.layers.nms import multiclass_nms


class RotatedBBoxHead(nn.Module):
    """
    Minimal rotated bbox head base class.

    This is the dependency-light counterpart of mmrotate's bbox head stack.
    It keeps only the parts needed by this project:
        - cls/reg output conventions
        - loss computation
        - proposal decoding and post-NMS inference
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        roi_feat_size: int = 7,
        reg_class_agnostic: bool = True,
        bbox_coder: Optional[object] = None,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_per_img: int = 100,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.roi_feat_size = int(roi_feat_size)
        self.reg_class_agnostic = bool(reg_class_agnostic)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.max_per_img = int(max_per_img)
        self.loss_cls_weight = float(loss_cls_weight)
        self.loss_bbox_weight = float(loss_bbox_weight)

        self.bbox_coder = bbox_coder or DeltaXYWHAOBBoxCoder()
        self.with_cls = True
        self.with_reg = True
        self.fc_cls: Optional[nn.Module] = None
        self.fc_reg: Optional[nn.Module] = None

    @property
    def roi_feat_area(self) -> int:
        return self.roi_feat_size * self.roi_feat_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def loss(
        self,
        cls_score: Optional[torch.Tensor],
        bbox_pred: Optional[torch.Tensor],
        rois: torch.Tensor,
        labels: torch.Tensor,
        label_weights: Optional[torch.Tensor],
        bbox_targets: Optional[torch.Tensor],
        bbox_weights: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        device = rois.device

        if label_weights is None:
            label_weights = torch.ones_like(labels, dtype=torch.float32, device=device)

        if cls_score is not None:
            per_sample_cls = F.cross_entropy(cls_score, labels, reduction="none")
            cls_denom = label_weights.sum().clamp(min=1.0)
            losses["loss_cls"] = self.loss_cls_weight * (per_sample_cls * label_weights).sum() / cls_denom
            with torch.no_grad():
                valid = label_weights > 0
                if valid.any():
                    pred = cls_score.argmax(dim=1)
                    losses["acc"] = (pred[valid] == labels[valid]).float().mean()
                else:
                    losses["acc"] = cls_score.new_tensor(0.0)

        if bbox_pred is not None:
            if bbox_targets is None:
                bbox_targets = bbox_pred.new_zeros((bbox_pred.size(0), 5))
            if bbox_weights is None:
                bbox_weights = bbox_pred.new_zeros((bbox_pred.size(0), 5))

            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), self.num_classes, 5)[
                        pos_inds, labels[pos_inds]
                    ]
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                loss_bbox = F.smooth_l1_loss(
                    pos_bbox_pred,
                    pos_bbox_targets,
                    reduction="none",
                )
                loss_bbox = (loss_bbox * pos_bbox_weights).sum() / pos_bbox_weights.sum().clamp(min=1.0)
                losses["loss_bbox"] = self.loss_bbox_weight * loss_bbox
            else:
                losses["loss_bbox"] = bbox_pred.sum() * 0.0

        return losses

    def get_bboxes(
        self,
        rois: torch.Tensor,
        cls_score: torch.Tensor,
        bbox_pred: Optional[torch.Tensor],
        img_shape,
        rescale: bool = False,
        cfg: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rescale
        if rois.numel() == 0:
            return rois.new_zeros((0, 6)), rois.new_zeros((0,), dtype=torch.long)

        test_cfg = cfg or {}
        score_thresh = float(test_cfg.get("score_thresh", self.score_thresh))
        nms_thresh = float(test_cfg.get("nms_thresh", self.nms_thresh))
        max_per_img = int(test_cfg.get("max_per_img", self.max_per_img))
        scores = F.softmax(cls_score, dim=1)

        fg_scores = scores[:, 1:] if scores.size(1) > 1 else scores[:, :0]
        if bbox_pred is None:
            bboxes = rois[:, 1:]
        elif self.reg_class_agnostic:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bbox_pred = bbox_pred.view(bbox_pred.size(0), self.num_classes, 5)
            per_class_boxes = []
            for class_idx in range(1, self.num_classes):
                per_class_boxes.append(
                    self.bbox_coder.decode(
                        rois[:, 1:],
                        bbox_pred[:, class_idx, :],
                        max_shape=img_shape,
                    )
                )
            bboxes = torch.cat(per_class_boxes, dim=1) if per_class_boxes else rois.new_zeros((rois.size(0), 0))

        det_bboxes, det_labels = multiclass_nms(
            boxes=bboxes,
            scores=fg_scores,
            score_threshold=score_thresh,
            iou_threshold=nms_thresh,
            max_num=max_per_img,
            mode="obb",
            obb_fallback_to_hbb=True,
        )
        return det_bboxes, det_labels
