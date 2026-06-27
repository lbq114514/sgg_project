from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.structures.boxes import BoxList


class ROIBoxHead(nn.Module):
    """
    Thin adapter around ``OrientedStandardRoIHead``.

    It exposes a maskrcnn-benchmark-like interface:
        ``x, detections, losses = head(features, proposals, targets)``.
    """

    def __init__(self, roi_head: nn.Module, num_classes: int):
        super().__init__()
        self.roi_head = roi_head
        self.num_classes = int(num_classes)

    def forward(
        self,
        features,
        proposals: Sequence[BoxList],
        targets=None,
        logger=None,
    ):
        del logger
        if self.training:
            if targets is None:
                raise ValueError("ROIBoxHead training requires targets.")
            bbox_results = self.roi_head.forward_train(features, proposals, targets)
            detections = list(bbox_results.get("sampled_boxes", proposals))
            self._attach_box_predictions(
                detections,
                bbox_results.get("cls_score"),
                bbox_results.get("bbox_pred"),
            )
            bbox_feats = bbox_results.get("bbox_feats")
            losses = bbox_results.get("loss_bbox", {})
            return bbox_feats, detections, losses

        det_bboxes, det_labels = self.roi_head.simple_test_bboxes(features, proposals)
        detections = self._to_boxlists(proposals, det_bboxes, det_labels)
        return None, detections, {}

    def _attach_box_predictions(self, proposals, cls_score, bbox_pred):
        if cls_score is None:
            return
        splits = [len(p) for p in proposals]
        cls_chunks = cls_score.split(splits, dim=0)
        bbox_chunks = bbox_pred.split(splits, dim=0) if bbox_pred is not None else [None] * len(proposals)

        for proposal, cls_chunk, bbox_chunk in zip(proposals, cls_chunks, bbox_chunks):
            if len(proposal) == 0:
                proposal.add_field("predict_logits", proposal.bbox.new_zeros((0, self.num_classes)))
                proposal.add_field("pred_labels", torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device))
                proposal.add_field("pred_scores", proposal.bbox.new_zeros((0,)))
                proposal.add_field("boxes_per_cls", proposal.bbox.new_zeros((0, self.num_classes, 5)))
                continue

            scores = F.softmax(cls_chunk, dim=1)
            pred_scores, fg_labels = scores[:, 1:].max(dim=1)
            pred_labels = fg_labels + 1
            proposal.add_field("predict_logits", cls_chunk)
            proposal.add_field("pred_labels", pred_labels)
            proposal.add_field("pred_scores", pred_scores)

            if bbox_chunk is None:
                boxes_per_cls = proposal.bbox[:, None, :].expand(-1, self.num_classes, -1)
            elif self.roi_head.bbox_head.reg_class_agnostic:
                decoded = self.roi_head.bbox_head.bbox_coder.decode(proposal.bbox, bbox_chunk)
                boxes_per_cls = decoded[:, None, :].expand(-1, self.num_classes, -1)
            else:
                decoded = []
                bbox_chunk = bbox_chunk.view(bbox_chunk.size(0), self.num_classes, 5)
                for class_idx in range(self.num_classes):
                    decoded.append(
                        self.roi_head.bbox_head.bbox_coder.decode(
                            proposal.bbox,
                            bbox_chunk[:, class_idx, :],
                        )
                    )
                boxes_per_cls = torch.stack(decoded, dim=1)
            proposal.add_field("boxes_per_cls", boxes_per_cls)

    def _to_boxlists(self, proposals, det_bboxes, det_labels):
        outputs = []
        for proposal, boxes, labels in zip(proposals, det_bboxes, det_labels):
            box_tensor = boxes[:, :5] if boxes.numel() > 0 else proposal.bbox.new_zeros((0, 5))
            result = BoxList(box_tensor, proposal.size, "xywha")
            pred_labels = labels + 1 if labels.numel() > 0 else labels
            pred_scores = boxes[:, 5] if boxes.numel() > 0 else proposal.bbox.new_zeros((0,))
            predict_logits = proposal.bbox.new_zeros((len(result), self.num_classes))
            if pred_labels.numel() > 0:
                predict_logits[torch.arange(len(result), device=pred_labels.device), pred_labels.clamp(max=self.num_classes - 1)] = pred_scores
            result.add_field("pred_labels", pred_labels)
            result.add_field("pred_scores", pred_scores)
            result.add_field("predict_logits", predict_logits)
            result.add_field("boxes_per_cls", box_tensor[:, None, :].expand(-1, self.num_classes, -1))
            outputs.append(result)
        return outputs
