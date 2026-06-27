from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.data.collate import pad_images
from sgg.modeling.anchor_generator.anchor_generator import AnchorGenerator
from sgg.modeling.box_coder.box_coder import UnifiedBoxCoder
from sgg.modeling.matcher.matcher import Matcher
from sgg.modeling.layers.nms import batched_nms
from sgg.modeling.backbone.swin import SwinTransformer
from sgg.modeling.box_coder.box_coder import DeltaXYWHAOBBoxCoder
from sgg.modeling.heads import RotatedShared2FCBBoxHead
from sgg.modeling.heads.rotated_rpn_head import RotatedRPNHead
from sgg.modeling.neck.fpn import FPN
from sgg.modeling.roi_extractors.rotated_roi_extractor import RotatedROIExtractor
from sgg.modeling.roi_heads.oriented_standard_roi_head import OrientedStandardRoIHead
from sgg.modeling.roi_heads import build_roi_heads
from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


def _to_onehot_logits(labels: torch.Tensor, num_classes: int, fill: float = 1000.0) -> torch.Tensor:
    logits = labels.new_full((labels.numel(), num_classes), -fill, dtype=torch.float32)
    if labels.numel() > 0:
        row_idx = torch.arange(labels.numel(), device=labels.device)
        logits[row_idx, labels.long()] = fill
    return logits


class SceneGraphDetector(nn.Module):
    """
    A lightweight but image-grounded SGG detector.

    The detector has three pieces of real model logic:
    - a CNN feature extractor
    - a dense detection head for `sgdet`
    - ROI-based object/node features for relation prediction
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["MODEL"]
        rel_cfg = model_cfg["RELATION_HEAD"]

        self.task = model_cfg["TASK"]
        self.box_mode = model_cfg["BOX_MODE"]
        self.num_classes = int(model_cfg["NUM_CLASSES"])
        self.num_predicates = int(model_cfg["NUM_PREDICATES"])
        self.hidden_dim = int(rel_cfg["NODE_DIM"])
        self.box_dim = 4 if self.box_mode == "hbb" else 5
        self.use_combined_roi_heads = bool(model_cfg.get("USE_COMBINED_ROI_HEADS", True))
        self.backbone_name = model_cfg.get("BACKBONE", {}).get("NAME", "simple_cnn")
        self.neck_name = model_cfg.get("NECK", {}).get("NAME", "")

        roi_cfg = model_cfg.get("ROI_EXTRACTOR", {})
        self.max_proposals = int(model_cfg.get("PROPOSAL_GENERATOR", {}).get("NUM_PROPOSALS", 64))
        self.pre_nms_topk = int(model_cfg.get("PROPOSAL_GENERATOR", {}).get("PRE_NMS_TOPK", self.max_proposals * 4))
        self.score_thresh = float(model_cfg.get("PROPOSAL_GENERATOR", {}).get("SCORE_THRESH", 0.05))
        self.nms_thresh = float(model_cfg.get("PROPOSAL_GENERATOR", {}).get("NMS_THRESH", 0.5))
        self.feature_stride = int(roi_cfg.get("FEATURE_STRIDE", 4))

        self.backbone, backbone_channels, backbone_strides = self._build_backbone(model_cfg.get("BACKBONE", {}))
        self.neck, neck_channels = self._build_neck(model_cfg.get("NECK", {}), backbone_channels)
        self.feature_channels = neck_channels if self.neck is not None else backbone_channels[-1]
        feature_strides = self._infer_feature_strides(backbone_strides, model_cfg.get("NECK", {}))
        self.feature_keys = [f"p{i + 2}" for i in range(len(feature_strides))]
        default_feature_key = roi_cfg.get("FEATURE_KEY", self.feature_keys[0])
        if default_feature_key not in self.feature_keys:
            default_feature_key = self.feature_keys[0]
        self.det_feature_key = model_cfg.get("DETECTION_FEATURE_KEY", default_feature_key)
        if self.det_feature_key not in self.feature_keys:
            self.det_feature_key = self.feature_keys[0]
        self.roi_feature_key = default_feature_key
        self.feature_stride_map = dict(zip(self.feature_keys, feature_strides))
        self.feature_stride = int(self.feature_stride_map[self.roi_feature_key])
        test_cfg = cfg.get("TEST", {})
        self.patch_auto_enabled = bool(test_cfg.get("PATCH_AUTO_ENABLED", True))
        self.patch_auto_min_size = int(test_cfg.get("PATCH_AUTO_MIN_SIZE", 1024))
        self.patch_max_pyramid_layers = int(test_cfg.get("PATCH_MAX_PYRAMID_LAYERS", 8))
        self.patch_batch_size = int(test_cfg.get("PATCH_BATCH_SIZE", 2))
        self.patch_batch_size_large = int(test_cfg.get("PATCH_BATCH_SIZE_LARGE", 4))
        self.patch_gaps = list(test_cfg.get("PATCH_GAPS", [200]))
        self.patch_size = tuple(int(v) for v in test_cfg.get("PATCH_SIZE", [1024, 1024]))
        self.patch_score_thresholds = [float(v) for v in test_cfg.get("PATCH_SCORE_THRESHOLDS", [0.3, 0.2, 0.1, 0.001, 0.00001])]

        proposal_cfg = model_cfg.get("PROPOSAL_GENERATOR", {})
        anchor_sizes = proposal_cfg.get("SIZES")
        if anchor_sizes is None:
            anchor_sizes = ((self.feature_stride * 8,),)
        else:
            anchor_sizes = tuple(tuple(level) for level in anchor_sizes)
        anchor_ratios = tuple(proposal_cfg.get("ASPECT_RATIOS", (0.5, 1.0, 2.0)))
        anchor_angles = tuple(proposal_cfg.get("ANGLES", (0.0,)))
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
            strides=(self.feature_stride_map[self.det_feature_key],),
            angles=anchor_angles,
            mode=self.box_mode,
        )
        self.rpn_num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        rpn_cfg = model_cfg.get("RPN_HEAD", {})
        self.rpn_head = None
        if rpn_cfg.get("NAME", "") == "oriented_rpn_head":
            self.rpn_head = RotatedRPNHead(
                in_channels=int(rpn_cfg.get("IN_CHANNELS", self.feature_channels)),
                feat_channels=int(rpn_cfg.get("FEAT_CHANNELS", self.feature_channels)),
                num_anchors=int(rpn_cfg.get("NUM_ANCHORS", 3)),
                cls_out_channels=int(rpn_cfg.get("CLS_OUT_CHANNELS", 1)),
                version=rpn_cfg.get("VERSION", "le90"),
                use_sigmoid_cls=bool(rpn_cfg.get("USE_SIGMOID_CLS", True)),
            )
        self.manual_rpn = self.rpn_head is None
        self.det_tower = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.rpn_objectness_head = nn.Conv2d(self.hidden_dim, self.rpn_num_anchors, kernel_size=1)
        self.rpn_box_head = nn.Conv2d(self.hidden_dim, self.rpn_num_anchors * self.box_dim, kernel_size=1)
        self.rpn_box_coder = UnifiedBoxCoder(
            mode=self.box_mode,
            angle_normalize=False,
        ) if self.box_mode == "obb" else UnifiedBoxCoder(mode="hbb")
        self.rpn_matcher = Matcher(
            high_threshold=float(proposal_cfg.get("FG_IOU_THRESHOLD", 0.7)),
            low_threshold=float(proposal_cfg.get("BG_IOU_THRESHOLD", 0.3)),
            allow_low_quality_matches=True,
        )
        self.obb_fallback_to_hbb = bool(proposal_cfg.get("OBB_FALLBACK_TO_HBB", True))
        self.roi_heads = build_roi_heads(cfg, self.feature_channels) if self.use_combined_roi_heads else None
        self.roi_head = self._build_standalone_roi_head(model_cfg, self.feature_channels)
        pretrained = model_cfg.get("PRETRAINED_DETECTOR", "")
        if pretrained:
            self.load_detector_pretrained(pretrained)

    def forward(self, images: torch.Tensor, targets: Optional[Sequence[BoxList]] = None):
        if images.ndim != 4:
            raise ValueError("images must have shape (N, C, H, W)")

        features = self._extract_features(images)

        if self.training:
            if targets is None:
                raise ValueError("Training requires targets.")
            proposals, det_losses = self._prepare_train_proposals(features, images, targets)
            if self.roi_heads is None or "relation" not in self.roi_heads:
                return det_losses
            _, _, rel_losses = self.roi_heads.relation(
                features,
                proposals,
                targets,
                OBj=self,
            )
            return {**det_losses, **rel_losses}

        if self.task == "sgdet" and self.patch_auto_enabled and self._should_use_patch_inference(images):
            proposals = self._prepare_eval_proposals_with_patches(images)
        else:
            proposals = self._prepare_eval_proposals(features, images, targets)
        if self.roi_heads is None or "relation" not in self.roi_heads:
            return proposals
        _, result, _ = self.roi_heads.relation(
            features,
            proposals,
            targets if self.task != "sgdet" else None,
            OBj=self,
        )
        return result

    def bbox_roi_extractor(self, features, proposals):
        if self.roi_head is None:
            raise RuntimeError("roi_head is not initialized.")
        return self.roi_head.bbox_roi_extractor(features, proposals)

    def bbox_head(self, roi_feats: torch.Tensor) -> torch.Tensor:
        if self.roi_head is None:
            raise RuntimeError("roi_head is not initialized.")
        return self._forward_bbox_features(roi_feats)

    def _forward_train(
        self,
        features: Dict[str, torch.Tensor],
        images: torch.Tensor,
        targets: Sequence[BoxList],
    ) -> Tuple[List[BoxList], List[torch.Tensor], Dict[str, torch.Tensor]]:
        feat = features[self.det_feature_key]
        rpn_objectness, rpn_box_deltas = self._rpn_forward(features, feat)

        det_losses = self._rpn_losses(
            objectness=rpn_objectness,
            box_deltas=rpn_box_deltas,
            targets=targets,
            image_size=(images.shape[-2], images.shape[-1]),
        )

        proposals: List[BoxList] = []
        node_feats: List[torch.Tensor] = []
        for batch_idx, target in enumerate(targets):
            if self.task == "predcls":
                proposal = self._copy_boxlist(target)
                labels = proposal.get_field("labels").long()
                predict_logits = _to_onehot_logits(
                    labels.clamp(min=0, max=self.num_classes - 1),
                    num_classes=self.num_classes,
                )
                self._attach_object_fields(
                    proposal,
                    gt_labels=labels,
                    predict_logits=predict_logits.to(proposal.bbox.device),
                    pred_labels=labels,
                    pred_scores=torch.ones(len(proposal), device=proposal.bbox.device),
                )
                roi_feat = self._extract_roi_node_features(features, [proposal], [batch_idx])[0]
                node_feat = roi_feat + self.label_embed(proposal.get_field("labels").long().clamp(min=0, max=self.num_classes - 1))
            elif self.task == "sgcls":
                proposal = self._copy_boxlist(target)
                roi_feat = self._extract_roi_node_features(features, [proposal], [batch_idx])[0]
                cls_logits = self.cls_head_roi(roi_feat)
                gt_labels = target.get_field("labels").long().clamp(min=0, max=self.num_classes - 1)
                cls_loss = F.cross_entropy(cls_logits, gt_labels)
                det_losses["loss_obj_cls"] = det_losses["loss_obj_cls"] + cls_loss
                pred_labels, pred_scores = self._decode_object_logits(cls_logits)
                self._attach_object_fields(
                    proposal,
                    gt_labels=gt_labels,
                    predict_logits=cls_logits,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                )
                node_feat = roi_feat + self.label_embed(pred_labels.clamp(min=0, max=self.num_classes - 1))
            else:
                proposal = self._decode_rpn_proposals(
                    objectness=rpn_objectness[batch_idx],
                    box_deltas=rpn_box_deltas[batch_idx],
                    image_size=(images.shape[-2], images.shape[-1]),
                )
                matched = self._align_proposals_to_gt(proposal, target)
                roi_feat = self._extract_roi_node_features(features, [proposal], [batch_idx])[0]
                if len(proposal) > 0:
                    cls_logits = self.cls_head_roi(roi_feat)
                    refine_delta = self.box_refine_head(roi_feat)
                    gt_labels = matched.get_field("labels").long()
                    pos_mask = gt_labels > 0
                    if pos_mask.any():
                        det_losses["loss_obj_cls"] = det_losses["loss_obj_cls"] + F.cross_entropy(
                            cls_logits[pos_mask],
                            gt_labels[pos_mask].clamp(min=0, max=self.num_classes - 1),
                        )
                        refined = self._refine_boxes(proposal.bbox, refine_delta, proposal.size)
                        det_losses["loss_box_reg"] = det_losses["loss_box_reg"] + F.smooth_l1_loss(
                            refined[pos_mask],
                            matched.bbox[pos_mask],
                        )
                    pred_labels, pred_scores = self._decode_object_logits(cls_logits)
                    proposal.bbox = self._refine_boxes(proposal.bbox, refine_delta, proposal.size)
                    self._attach_object_fields(
                        proposal,
                        gt_labels=gt_labels,
                        predict_logits=cls_logits,
                        pred_labels=pred_labels,
                        pred_scores=pred_scores,
                    )
                    self._seed_relation_targets_from_matches(proposal, matched)
                    node_feat = roi_feat + self.label_embed(pred_labels.clamp(min=0, max=self.num_classes - 1))
                else:
                    proposal.add_field("gt_labels", torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device))
                    proposal.add_field("pair_labels", torch.zeros((0, 0), dtype=torch.long, device=proposal.bbox.device))
                    proposal.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=proposal.bbox.device))
                    node_feat = roi_feat

            proposals.append(proposal)
            node_feats.append(node_feat)

        if self.task == "sgcls":
            det_losses["loss_obj_cls"] = det_losses["loss_obj_cls"] / max(len(targets), 1)
        if self.task == "sgdet":
            det_losses["loss_obj_cls"] = det_losses["loss_obj_cls"] / max(len(targets), 1)
            det_losses["loss_box_reg"] = det_losses["loss_box_reg"] / max(len(targets), 1)
        return proposals, node_feats, det_losses

    def _zero_detection_losses(self, device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=device)
        return {
            "loss_obj_cls": zero.clone(),
            "loss_box_reg": zero.clone(),
            "loss_objectness": zero.clone(),
        }

    def _prepare_train_proposals(
        self,
        features: Dict[str, torch.Tensor],
        images: torch.Tensor,
        targets: Sequence[BoxList],
    ) -> Tuple[List[BoxList], Dict[str, torch.Tensor]]:
        det_losses = self._zero_detection_losses(images.device)

        if self.task == "predcls":
            proposals = [self._copy_boxlist(target) for target in targets]
            for proposal in proposals:
                labels = proposal.get_field("labels").long()
                predict_logits = _to_onehot_logits(
                    labels.clamp(min=0, max=self.num_classes - 1),
                    num_classes=self.num_classes,
                )
                self._attach_object_fields(
                    proposal,
                    gt_labels=labels,
                    predict_logits=predict_logits.to(proposal.bbox.device),
                    pred_labels=labels,
                    pred_scores=torch.ones(len(proposal), device=proposal.bbox.device),
                )
            return proposals, det_losses

        if self.task == "sgcls":
            proposals = [self._copy_boxlist(target) for target in targets]
            cls_chunks, _ = self._run_standalone_bbox_head(features, proposals)
            for proposal, target, cls_chunk in zip(proposals, targets, cls_chunks):
                gt_labels = target.get_field("labels").long().clamp(min=0, max=self.num_classes - 1)
                if cls_chunk.numel() > 0:
                    det_losses["loss_obj_cls"] = det_losses["loss_obj_cls"] + F.cross_entropy(cls_chunk, gt_labels)
                    pred_labels, pred_scores = self._decode_object_logits(cls_chunk)
                else:
                    pred_labels = torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device)
                    pred_scores = torch.zeros((0,), dtype=torch.float32, device=proposal.bbox.device)
                self._attach_object_fields(
                    proposal,
                    gt_labels=gt_labels,
                    predict_logits=cls_chunk,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                )
            det_losses["loss_obj_cls"] = det_losses["loss_obj_cls"] / max(len(targets), 1)
            return proposals, det_losses

        feat = features[self.det_feature_key]
        rpn_objectness, rpn_box_deltas = self._rpn_forward(features, feat)
        det_losses = self._rpn_losses(
            objectness=rpn_objectness,
            box_deltas=rpn_box_deltas,
            targets=targets,
            image_size=(images.shape[-2], images.shape[-1]),
        )
        proposals = [
            self._decode_rpn_proposals(
                objectness=rpn_objectness[batch_idx],
                box_deltas=rpn_box_deltas[batch_idx],
                image_size=(images.shape[-2], images.shape[-1]),
            )
            for batch_idx in range(images.size(0))
        ]
        bbox_results = self.roi_head.forward_train(features, proposals, targets)
        detections = list(bbox_results.get("sampled_boxes", proposals))
        labels_all = bbox_results.get("labels")
        self._attach_box_predictions(
            detections,
            bbox_results.get("cls_score"),
            bbox_results.get("bbox_pred"),
            labels_all=labels_all,
        )
        bbox_loss = bbox_results.get("loss_bbox", {})
        for key, value in bbox_loss.items():
            det_losses[key] = value
        return detections, det_losses

    def _prepare_eval_proposals(
        self,
        features: Dict[str, torch.Tensor],
        images: torch.Tensor,
        targets: Optional[Sequence[BoxList]],
    ) -> List[BoxList]:
        if self.task == "predcls":
            if targets is None:
                raise ValueError("predcls evaluation requires targets.")
            proposals = [self._copy_boxlist(target) for target in targets]
            for proposal in proposals:
                labels = proposal.get_field("labels").long()
                predict_logits = _to_onehot_logits(
                    labels.clamp(min=0, max=self.num_classes - 1),
                    num_classes=self.num_classes,
                )
                self._attach_object_fields(
                    proposal,
                    gt_labels=labels,
                    predict_logits=predict_logits.to(proposal.bbox.device),
                    pred_labels=labels,
                    pred_scores=torch.ones(len(proposal), device=proposal.bbox.device),
                )
            return proposals

        if self.task == "sgcls":
            if targets is None:
                raise ValueError("sgcls evaluation requires targets.")
            proposals = [self._copy_boxlist(target) for target in targets]
            cls_chunks, _ = self._run_standalone_bbox_head(features, proposals)
            for proposal, target, cls_chunk in zip(proposals, targets, cls_chunks):
                pred_labels, pred_scores = self._decode_object_logits(cls_chunk)
                self._attach_object_fields(
                    proposal,
                    gt_labels=target.get_field("labels").long().clamp(min=0, max=self.num_classes - 1),
                    predict_logits=cls_chunk,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                )
                proposal.add_field("labels", pred_labels)
            return proposals

        return self._detect_sgdet_boxes(features, images.shape[-2:], images.device)

    def _run_standalone_bbox_head(
        self,
        features: Dict[str, torch.Tensor],
        proposals: Sequence[BoxList],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.roi_head is None:
            raise RuntimeError("roi_head is not initialized.")
        if not proposals:
            return [], []
        total_boxes = sum(len(p) for p in proposals)
        if total_boxes == 0:
            empty_logits = [features[self.roi_feature_key].new_zeros((0, self.num_classes)) for _ in proposals]
            empty_bbox = [features[self.roi_feature_key].new_zeros((0, 5)) for _ in proposals]
            return empty_logits, empty_bbox

        bbox_feats = self.bbox_roi_extractor(features, list(proposals))
        cls_score, bbox_pred = self.roi_head.bbox_head(bbox_feats)
        splits = [len(p) for p in proposals]
        cls_chunks = list(cls_score.split(splits, dim=0))
        bbox_chunks = list(bbox_pred.split(splits, dim=0)) if bbox_pred is not None else [
            features[self.roi_feature_key].new_zeros((len(p), 5)) for p in proposals
        ]
        return cls_chunks, bbox_chunks

    def _forward_bbox_features(self, roi_feats: torch.Tensor) -> torch.Tensor:
        head = self.roi_head.bbox_head
        x = roi_feats
        if x.dim() > 2:
            x = x.flatten(1)
        for fc in head.shared_fcs:
            x = head.relu(fc(x))
        return x

    def _attach_object_fields(
        self,
        proposal: BoxList,
        *,
        gt_labels: Optional[torch.Tensor] = None,
        predict_logits: Optional[torch.Tensor] = None,
        pred_labels: Optional[torch.Tensor] = None,
        pred_scores: Optional[torch.Tensor] = None,
    ) -> None:
        if gt_labels is not None:
            gt_labels = gt_labels.long()
            proposal.add_field("gt_labels", gt_labels)
            proposal.add_field("labels", gt_labels)
        if predict_logits is not None:
            proposal.add_field("predict_logits", predict_logits)
            proposal.add_field("pred_logits", predict_logits)
        if pred_labels is not None:
            proposal.add_field("pred_labels", pred_labels.long())
        if pred_scores is not None:
            proposal.add_field("pred_scores", pred_scores)
            proposal.add_field("scores", pred_scores)

    def _attach_box_predictions(
        self,
        proposals: Sequence[BoxList],
        cls_score: Optional[torch.Tensor],
        bbox_pred: Optional[torch.Tensor],
        labels_all: Optional[torch.Tensor] = None,
    ) -> None:
        if cls_score is None:
            return
        splits = [len(p) for p in proposals]
        cls_chunks = cls_score.split(splits, dim=0)
        bbox_chunks = bbox_pred.split(splits, dim=0) if bbox_pred is not None else [None] * len(proposals)
        label_chunks = labels_all.split(splits, dim=0) if labels_all is not None else [None] * len(proposals)
        for proposal, cls_chunk, bbox_chunk, label_chunk in zip(proposals, cls_chunks, bbox_chunks, label_chunks):
            if len(proposal) == 0:
                proposal.add_field("predict_logits", proposal.bbox.new_zeros((0, self.num_classes)))
                proposal.add_field("pred_logits", proposal.bbox.new_zeros((0, self.num_classes)))
                proposal.add_field("pred_labels", torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device))
                proposal.add_field("pred_scores", proposal.bbox.new_zeros((0,)))
                proposal.add_field("scores", proposal.bbox.new_zeros((0,)))
                proposal.add_field("boxes_per_cls", proposal.bbox.new_zeros((0, self.num_classes, 5)))
                proposal.add_field("labels", torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device))
                if label_chunk is not None:
                    proposal.add_field("gt_labels", torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device))
                continue
            probs = F.softmax(cls_chunk, dim=1)
            pred_scores, fg_labels = probs[:, 1:].max(dim=1)
            pred_labels = fg_labels + 1
            self._attach_object_fields(
                proposal,
                gt_labels=label_chunk.long() if label_chunk is not None else None,
                predict_logits=cls_chunk,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
            )
            if label_chunk is not None:
                proposal.add_field("labels", label_chunk.long())
            else:
                proposal.add_field("labels", pred_labels)
            if bbox_chunk is None:
                boxes_per_cls = proposal.bbox[:, None, :].expand(-1, self.num_classes, -1)
            elif self.roi_head.bbox_head.reg_class_agnostic:
                decoded = self.roi_head.bbox_head.bbox_coder.decode(proposal.bbox, bbox_chunk)
                boxes_per_cls = decoded[:, None, :].expand(-1, self.num_classes, -1)
                proposal.bbox = decoded
            else:
                boxes_per_cls = proposal.bbox[:, None, :].expand(-1, self.num_classes, -1)
            proposal.add_field("boxes_per_cls", boxes_per_cls)

    def _forward_eval(
        self,
        features: Dict[str, torch.Tensor],
        images: torch.Tensor,
        targets: Optional[Sequence[BoxList]],
    ) -> Tuple[List[BoxList], List[torch.Tensor]]:
        feat = features[self.det_feature_key]
        rpn_objectness, rpn_box_deltas = self._rpn_forward(features, feat)

        proposals: List[BoxList] = []
        node_feats: List[torch.Tensor] = []

        if self.task in {"predcls", "sgcls"}:
            if targets is None:
                raise ValueError(f"{self.task} evaluation requires targets.")
            for batch_idx, target in enumerate(targets):
                proposal = self._copy_boxlist(target)
                roi_feat = self._extract_roi_node_features(features, [proposal], [batch_idx])[0]
                if self.task == "predcls":
                    labels = proposal.get_field("labels").long()
                    predict_logits = _to_onehot_logits(
                        labels.clamp(min=0, max=self.num_classes - 1),
                        num_classes=self.num_classes,
                    )
                    self._attach_object_fields(
                        proposal,
                        gt_labels=labels,
                        predict_logits=predict_logits.to(proposal.bbox.device),
                        pred_labels=labels,
                        pred_scores=torch.ones(len(proposal), device=proposal.bbox.device),
                    )
                    node_feat = roi_feat + self.label_embed(labels.clamp(min=0, max=self.num_classes - 1))
                else:
                    cls_logits = self.cls_head_roi(roi_feat)
                    pred_labels, pred_scores = self._decode_object_logits(cls_logits)
                    self._attach_object_fields(
                        proposal,
                        gt_labels=target.get_field("labels").long().clamp(min=0, max=self.num_classes - 1),
                        predict_logits=cls_logits,
                        pred_labels=pred_labels,
                        pred_scores=pred_scores,
                    )
                    proposal.add_field("labels", pred_labels)
                    node_feat = roi_feat + self.label_embed(pred_labels.clamp(min=0, max=self.num_classes - 1))
                proposals.append(proposal)
                node_feats.append(node_feat)
            return proposals, node_feats

        for batch_idx in range(images.size(0)):
            proposal = self._decode_rpn_proposals(
                objectness=rpn_objectness[batch_idx],
                box_deltas=rpn_box_deltas[batch_idx],
                image_size=(images.shape[-2], images.shape[-1]),
            )
            roi_feat = self._extract_roi_node_features(features, [proposal], [batch_idx])[0]
            if len(proposal) > 0:
                cls_logits = self.cls_head_roi(roi_feat)
                refine_delta = self.box_refine_head(roi_feat)
                pred_labels, pred_scores = self._decode_object_logits(cls_logits)
                proposal.bbox = self._refine_boxes(proposal.bbox, refine_delta, proposal.size)
                self._attach_object_fields(
                    proposal,
                    predict_logits=cls_logits,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                )
                proposal.add_field("labels", pred_labels)
                node_feat = roi_feat + self.label_embed(pred_labels.clamp(min=0, max=self.num_classes - 1))
            else:
                node_feat = roi_feat
            proposals.append(proposal)
            node_feats.append(node_feat)
        return proposals, node_feats

    def _detect_sgdet_boxes(
        self,
        features: Dict[str, torch.Tensor],
        image_hw: Tuple[int, int],
        device: torch.device,
    ) -> List[BoxList]:
        feat = features[self.det_feature_key]
        rpn_objectness, rpn_box_deltas = self._rpn_forward(features, feat)
        proposals = [
            self._decode_rpn_proposals(
                objectness=rpn_objectness[batch_idx],
                box_deltas=rpn_box_deltas[batch_idx],
                image_size=image_hw,
            )
            for batch_idx in range(feat.size(0))
        ]
        det_bboxes, det_labels = self.roi_head.simple_test_bboxes(features, proposals)
        return self._detections_to_boxlists(proposals, det_bboxes, det_labels, device=device)

    def _detections_to_boxlists(
        self,
        proposals: Sequence[BoxList],
        det_bboxes: Sequence[torch.Tensor],
        det_labels: Sequence[torch.Tensor],
        device: torch.device,
    ) -> List[BoxList]:
        outputs = []
        for proposal, boxes, labels in zip(proposals, det_bboxes, det_labels):
            box_tensor = boxes[:, :5] if boxes.numel() > 0 else proposal.bbox.new_zeros((0, 5))
            result = BoxList(box_tensor, proposal.size, "xywha")
            pred_labels = labels + 1 if labels.numel() > 0 else labels
            pred_scores = boxes[:, 5] if boxes.numel() > 0 else proposal.bbox.new_zeros((0,))
            predict_logits = proposal.bbox.new_zeros((len(result), self.num_classes))
            if pred_labels.numel() > 0:
                predict_logits[torch.arange(len(result), device=pred_labels.device), pred_labels.clamp(max=self.num_classes - 1)] = pred_scores
            result.add_field("pred_labels", pred_labels.long())
            result.add_field("pred_scores", pred_scores)
            result.add_field("scores", pred_scores)
            result.add_field("predict_logits", predict_logits.to(device=device))
            result.add_field("pred_logits", predict_logits.to(device=device))
            result.add_field("labels", pred_labels.long())
            result.add_field("pair_labels", torch.zeros((len(result), len(result)), dtype=torch.long, device=device))
            result.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=device))
            outputs.append(result.to(device))
        return outputs

    def _should_use_patch_inference(self, images: torch.Tensor) -> bool:
        if self.task != "sgdet":
            return False
        _, _, h, w = images.shape
        return max(h, w) > self.patch_auto_min_size

    def _prepare_eval_proposals_with_patches(self, images: torch.Tensor) -> List[BoxList]:
        proposals = []
        for image in images:
            proposals.append(self._detect_single_image_multiscale(image))
        return proposals

    def _detect_single_image_multiscale(self, image: torch.Tensor) -> BoxList:
        _, h, w = image.shape
        scale_shapes = []
        cur_h, cur_w = h, w
        for _ in range(self.patch_max_pyramid_layers):
            scale_shapes.append((cur_h, cur_w))
            if max(cur_h, cur_w) <= self.patch_auto_min_size:
                break
            cur_h = max(cur_h // 2, 1)
            cur_w = max(cur_w // 2, 1)
        all_detections: List[BoxList] = []
        for scaled_h, scaled_w in scale_shapes:
            if scaled_h == h and scaled_w == w:
                scaled = image
            else:
                scaled = F.interpolate(image.unsqueeze(0), size=(scaled_h, scaled_w), mode="bilinear", align_corners=False).squeeze(0)
            det = self._detect_single_scale_patches(scaled, original_size=(w, h))
            if len(det) > 0:
                all_detections.append(det)
        if not all_detections:
            return BoxList(image.new_zeros((0, 5)), (w, h), "xywha")
        return self._merge_patch_detections(all_detections, (w, h))

    def _detect_single_scale_patches(self, image: torch.Tensor, original_size: Tuple[int, int]) -> BoxList:
        _, scaled_h, scaled_w = image.shape
        patch_h, patch_w = self.patch_size
        step_h = max(patch_h - max(self.patch_gaps), 1)
        step_w = max(patch_w - max(self.patch_gaps), 1)
        ys = self._sliding_starts(scaled_h, patch_h, step_h)
        xs = self._sliding_starts(scaled_w, patch_w, step_w)
        patches = []
        metas = []
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + patch_h, scaled_h)
                x1 = min(x0 + patch_w, scaled_w)
                patch = image[:, y0:y1, x0:x1]
                patches.append(patch)
                metas.append({"x0": x0, "y0": y0, "width": x1 - x0, "height": y1 - y0})
        detections: List[BoxList] = []
        batch_size = self.patch_batch_size_large if max(original_size[0], original_size[1]) <= 10000 else self.patch_batch_size
        for start in range(0, len(patches), batch_size):
            batch_patches = patches[start : start + batch_size]
            batch_metas = metas[start : start + batch_size]
            batch_tensor = pad_images(batch_patches, size_divisible=32).to(image.device)
            batch_features = self._extract_features(batch_tensor)
            patch_dets = self._detect_sgdet_boxes(
                batch_features,
                image_hw=(batch_tensor.shape[-2], batch_tensor.shape[-1]),
                device=image.device,
            )
            for det, meta in zip(patch_dets, batch_metas):
                detections.append(self._relocate_patch_detection(det, meta, (scaled_w, scaled_h), original_size))
        if not detections:
            return BoxList(image.new_zeros((0, 5)), original_size, "xywha")
        return self._merge_patch_detections(detections, original_size)

    def _relocate_patch_detection(
        self,
        det: BoxList,
        meta: Dict[str, int],
        scaled_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> BoxList:
        boxes = det.bbox.clone()
        if boxes.numel() > 0:
            boxes[:, 0] += float(meta["x0"])
            boxes[:, 1] += float(meta["y0"])
            sx = float(original_size[0]) / float(max(scaled_size[0], 1))
            sy = float(original_size[1]) / float(max(scaled_size[1], 1))
            boxes[:, 0] *= sx
            boxes[:, 1] *= sy
            boxes[:, 2] *= sx
            boxes[:, 3] *= sy
        relocated = BoxList(boxes, original_size, det.mode)
        for field in det.fields():
            value = det.get_field(field)
            if torch.is_tensor(value):
                value = value.clone()
            relocated.add_field(field, value)
        return relocated

    def _merge_patch_detections(self, detections: Sequence[BoxList], size: Tuple[int, int]) -> BoxList:
        boxes = torch.cat([det.bbox for det in detections], dim=0) if detections else torch.zeros((0, 5))
        pred_scores = torch.cat([det.get_field("pred_scores") for det in detections], dim=0) if detections else torch.zeros((0,))
        pred_labels = torch.cat([det.get_field("pred_labels") for det in detections], dim=0) if detections else torch.zeros((0,), dtype=torch.long)
        predict_logits = torch.cat([det.get_field("predict_logits") for det in detections], dim=0) if detections else torch.zeros((0, self.num_classes))
        if boxes.numel() == 0:
            empty = BoxList(boxes.new_zeros((0, 5)), size, "xywha")
            empty.add_field("pred_labels", pred_labels)
            empty.add_field("pred_scores", pred_scores)
            empty.add_field("scores", pred_scores)
            empty.add_field("predict_logits", predict_logits)
            empty.add_field("pred_logits", predict_logits)
            empty.add_field("labels", pred_labels)
            empty.add_field("pair_labels", torch.zeros((0, 0), dtype=torch.long, device=boxes.device))
            empty.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=boxes.device))
            return empty
        keep = None
        for thresh in self.patch_score_thresholds:
            cur_keep = torch.nonzero(pred_scores >= thresh, as_tuple=False).flatten()
            if cur_keep.numel() > 0:
                keep = cur_keep
                break
        if keep is None:
            keep = torch.arange(boxes.size(0), device=boxes.device)
        boxes = boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        predict_logits = predict_logits[keep]
        _, keep_nms = batched_nms(
            boxes,
            pred_scores,
            pred_labels,
            self.nms_thresh,
            mode="obb",
            obb_fallback_to_hbb=self.obb_fallback_to_hbb,
        )
        boxes = boxes[keep_nms]
        pred_scores = pred_scores[keep_nms]
        pred_labels = pred_labels[keep_nms]
        predict_logits = predict_logits[keep_nms]
        merged = BoxList(boxes, size, "xywha")
        merged.add_field("pred_labels", pred_labels.long())
        merged.add_field("pred_scores", pred_scores)
        merged.add_field("scores", pred_scores)
        merged.add_field("predict_logits", predict_logits)
        merged.add_field("pred_logits", predict_logits)
        merged.add_field("labels", pred_labels.long())
        merged.add_field("pair_labels", torch.zeros((len(merged), len(merged)), dtype=torch.long, device=boxes.device))
        merged.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=boxes.device))
        return merged

    def _sliding_starts(self, full: int, window: int, step: int) -> List[int]:
        if full <= window:
            return [0]
        starts = []
        pos = 0
        while True:
            starts.append(pos)
            if pos + window >= full:
                break
            pos = min(pos + step, full - window)
            if pos == starts[-1]:
                break
        return starts

    def _rpn_losses(
        self,
        objectness: torch.Tensor,
        box_deltas: torch.Tensor,
        targets: Sequence[BoxList],
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        device = objectness.device
        loss_obj = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        if self.task != "sgdet":
            return {
                "loss_obj_cls": loss_cls,
                "loss_box_reg": loss_box,
                "loss_objectness": loss_obj,
            }

        feat_h, feat_w = objectness.shape[-2:]
        anchors = self.anchor_generator.grid_priors([(feat_h, feat_w)], device=device)[0]
        for batch_idx, target in enumerate(targets):
            logits = objectness[batch_idx].permute(1, 2, 0).reshape(-1)
            deltas = box_deltas[batch_idx].permute(1, 2, 0).reshape(-1, self.box_dim)
            gt_boxes = target.bbox.float().to(device)

            cls_targets = torch.full((anchors.size(0),), -1.0, device=device)
            if gt_boxes.numel() == 0:
                cls_targets[:] = 0.0
                valid_mask = cls_targets >= 0
                loss_obj = loss_obj + F.binary_cross_entropy_with_logits(logits[valid_mask], cls_targets[valid_mask])
                continue

            iou = self._pairwise_iou(anchors, gt_boxes, target.size).transpose(0, 1)
            matches = self.rpn_matcher(iou)
            pos_mask = matches >= 0
            neg_mask = matches == Matcher.BELOW_LOW_THRESHOLD
            valid_mask = pos_mask | neg_mask
            cls_targets[neg_mask] = 0.0
            cls_targets[pos_mask] = 1.0
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(logits[valid_mask], cls_targets[valid_mask])

            if pos_mask.any():
                matched_gt = gt_boxes[matches[pos_mask]]
                target_deltas = self.rpn_box_coder.encode(anchors[pos_mask], matched_gt)
                loss_box = loss_box + F.smooth_l1_loss(deltas[pos_mask], target_deltas)

        divisor = max(len(targets), 1)
        return {
            "loss_obj_cls": loss_cls / divisor,
            "loss_box_reg": loss_box / divisor,
            "loss_objectness": loss_obj / divisor,
        }

    def _decode_rpn_proposals(
        self,
        objectness: torch.Tensor,
        box_deltas: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> BoxList:
        feat_h, feat_w = objectness.shape[-2:]
        anchors = self.anchor_generator.grid_priors([(feat_h, feat_w)], device=objectness.device)[0]
        scores = objectness.permute(1, 2, 0).reshape(-1).sigmoid()
        deltas = box_deltas.permute(1, 2, 0).reshape(-1, self.box_dim)
        keep_idx = torch.nonzero(scores > self.score_thresh, as_tuple=False).flatten()
        if keep_idx.numel() == 0:
            topk = min(self.max_proposals, scores.numel())
            keep_idx = scores.topk(topk).indices
        if keep_idx.numel() > self.pre_nms_topk:
            order = scores[keep_idx].topk(self.pre_nms_topk).indices
            keep_idx = keep_idx[order]

        proposal_scores = scores[keep_idx]
        boxes = self.rpn_box_coder.decode(anchors[keep_idx], deltas[keep_idx], max_shape=image_size)
        if self.box_mode == "obb":
            valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        else:
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        proposal_scores = proposal_scores[valid]
        if boxes.numel() == 0:
            img_h, img_w = image_size
            proposal = BoxList(boxes.new_zeros((0, self.box_dim)), (img_w, img_h), self._boxlist_mode())
            proposal.add_field("pred_scores", proposal_scores)
            proposal.add_field("scores", proposal_scores)
            proposal.add_field("pred_labels", torch.zeros((0,), dtype=torch.long, device=boxes.device))
            proposal.add_field("labels", torch.zeros((0,), dtype=torch.long, device=boxes.device))
            proposal.add_field("pair_labels", torch.zeros((0, 0), dtype=torch.long, device=boxes.device))
            proposal.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=boxes.device))
            return proposal

        labels = torch.zeros((boxes.size(0),), dtype=torch.long, device=boxes.device)
        _, nms_keep = batched_nms(
            boxes,
            proposal_scores,
            labels,
            self.nms_thresh,
            mode=self.box_mode,
            obb_fallback_to_hbb=self.obb_fallback_to_hbb,
        )
        nms_keep = nms_keep[: self.max_proposals]
        boxes = boxes[nms_keep]
        proposal_scores = proposal_scores[nms_keep]

        img_h, img_w = image_size
        proposal = BoxList(boxes, (img_w, img_h), self._boxlist_mode())
        proposal.add_field("pred_scores", proposal_scores)
        proposal.add_field("scores", proposal_scores)
        proposal.add_field("pred_labels", torch.zeros((len(proposal),), dtype=torch.long, device=boxes.device))
        proposal.add_field("labels", torch.zeros((len(proposal),), dtype=torch.long, device=boxes.device))
        proposal.add_field("pair_labels", torch.zeros((len(proposal), len(proposal)), dtype=torch.long, device=boxes.device))
        proposal.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=boxes.device))
        return proposal

    def _extract_roi_node_features(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[BoxList],
        batch_indices: List[int],
    ) -> List[torch.Tensor]:
        if len(proposals) == 0:
            return []
        feature_batch = {self.roi_feature_key: features[self.roi_feature_key][batch_indices]}
        pooled = self.roi_extractor(feature_batch, proposals)
        if pooled.numel() == 0:
            ref = features[self.roi_feature_key]
            return [ref.new_zeros((0, self.hidden_dim)) for _ in proposals]
        roi_feat = self.roi_head(pooled)
        splits = [len(p) for p in proposals]
        return list(roi_feat.split(splits, dim=0))

    def _rpn_forward(self, features: Dict[str, torch.Tensor], feat: torch.Tensor):
        if self.rpn_head is None or self.task == "sgdet":
            tower = self.det_tower(feat)
            return self.rpn_objectness_head(tower), self.rpn_box_head(tower)
        cls_scores, bbox_preds = self.rpn_head([features[key] for key in self.feature_keys])
        level_idx = self.feature_keys.index(self.det_feature_key)
        return cls_scores[level_idx], bbox_preds[level_idx]

    def _extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_feats = self.backbone(images)
        if isinstance(backbone_feats, torch.Tensor):
            backbone_feats = [backbone_feats]
        if self.neck is not None:
            feats = self.neck(backbone_feats)
        else:
            feats = backbone_feats
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        return {
            key: value
            for key, value in zip(self.feature_keys, feats)
        }

    def load_detector_pretrained(self, checkpoint_path: str):
        checkpoint_path = str(Path(checkpoint_path).expanduser())
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model_state = self.state_dict()
        loadable = {}
        skipped_d2 = 0
        loaded_keys = []
        missing_in_model = []
        shape_mismatches = []
        for key, value in state_dict.items():
            if any(part.endswith("d2") for part in key.split(".")):
                skipped_d2 += 1
                continue
            new_key = key
            if new_key.startswith("backbone.patch_embed.projection."):
                new_key = "backbone.patch_embed.proj." + new_key[len("backbone.patch_embed.projection."):]
            if new_key in model_state and model_state[new_key].shape == value.shape:
                loadable[new_key] = value
                loaded_keys.append(new_key)
            elif new_key not in model_state:
                missing_in_model.append((key, new_key))
            else:
                shape_mismatches.append((new_key, tuple(value.shape), tuple(model_state[new_key].shape)))
        missing, unexpected = self.load_state_dict(loadable, strict=False)
        print(
            f"Loaded detector pretrained weights from {checkpoint_path}: "
            f"{len(loadable)} tensors, skipped_d2={skipped_d2}, "
            f"skipped_unmatched={len(missing_in_model) + len(shape_mismatches)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        # print("Loaded parameter names:")
        # for key in sorted(loaded_keys):
        #     print(f"  [loaded] {key} shape={tuple(model_state[key].shape)}")
        # if missing_in_model:
        #     print("Checkpoint parameters not found in current model:")
        #     for original_key, mapped_key in sorted(missing_in_model):
        #         print(f"  [unmatched:not_in_model] checkpoint={original_key} mapped={mapped_key}")
        # if shape_mismatches:
        #     print("Checkpoint parameters with shape mismatch:")
        #     for key, ckpt_shape, model_shape in sorted(shape_mismatches):
        #         print(f"  [unmatched:shape] {key} checkpoint_shape={ckpt_shape} model_shape={model_shape}")
        # if missing:
        #     print("Current model parameters missing from checkpoint load:")
        #     for key in sorted(missing):
        #         print(f"  [missing] {key} shape={tuple(model_state[key].shape)}")
        # if unexpected:
        #     print("Unexpected parameter names returned by load_state_dict:")
        #     for key in sorted(unexpected):
        #         print(f"  [unexpected] {key}")

    def _build_backbone(self, backbone_cfg: Dict) -> Tuple[nn.Module, List[int], List[int]]:
        name = backbone_cfg.get("NAME", "simple_cnn")
        if name == "swin":
            embed_dims = int(backbone_cfg.get("EMBED_DIMS", 96))
            depths = tuple(backbone_cfg.get("DEPTHS", (2, 2, 6, 2)))
            num_heads = tuple(backbone_cfg.get("NUM_HEADS", (3, 6, 12, 24)))
            out_indices = tuple(backbone_cfg.get("OUT_INDICES", (0, 1, 2, 3)))
            patch_size = int(backbone_cfg.get("PATCH_SIZE", 4))
            backbone = SwinTransformer(
                in_channels=3,
                embed_dims=embed_dims,
                patch_size=patch_size,
                window_size=int(backbone_cfg.get("WINDOW_SIZE", 7)),
                mlp_ratio=float(backbone_cfg.get("MLP_RATIO", 4)),
                depths=depths,
                num_heads=num_heads,
                strides=tuple(backbone_cfg.get("STRIDES", (patch_size, 2, 2, 2))),
                out_indices=out_indices,
                qkv_bias=bool(backbone_cfg.get("QKV_BIAS", True)),
                qk_scale=backbone_cfg.get("QK_SCALE", None),
                patch_norm=bool(backbone_cfg.get("PATCH_NORM", True)),
                drop_rate=float(backbone_cfg.get("DROP_RATE", 0.0)),
                attn_drop_rate=float(backbone_cfg.get("ATTN_DROP_RATE", 0.0)),
                drop_path_rate=float(backbone_cfg.get("DROP_PATH_RATE", 0.1)),
                with_cp=bool(backbone_cfg.get("WITH_CP", False)),
            )
            channels = [backbone.num_features[i] for i in out_indices]
            strides = [patch_size * (2 ** i) for i in range(len(out_indices))]
            return backbone, channels, strides

        backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        return backbone, [self.hidden_dim], [4]

    def _build_neck(self, neck_cfg: Dict, backbone_channels: List[int]) -> Tuple[Optional[nn.Module], int]:
        name = neck_cfg.get("NAME", "")
        if name == "fpn_neck":
            in_channels = list(neck_cfg.get("IN_CHANNELS", backbone_channels))
            if len(in_channels) != len(backbone_channels):
                in_channels = list(backbone_channels)
            out_channels = int(neck_cfg.get("OUT_CHANNELS", self.hidden_dim))
            num_outs = int(neck_cfg.get("NUM_OUTS", len(in_channels)))
            neck = FPN(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                start_level=int(neck_cfg.get("START_LEVEL", 0)),
                end_level=int(neck_cfg.get("END_LEVEL", -1)),
                add_extra_convs=neck_cfg.get("ADD_EXTRA_CONVS", False),
                relu_before_extra_convs=bool(neck_cfg.get("RELU_BEFORE_EXTRA_CONVS", False)),
                no_norm_on_lateral=bool(neck_cfg.get("NO_NORM_ON_LATERAL", True)),
                upsample_mode=neck_cfg.get("UPSAMPLE_MODE", "nearest"),
                scale_factor=neck_cfg.get("SCALE_FACTOR", None),
            )
            return neck, out_channels
        return None, backbone_channels[-1]

    def _build_standalone_roi_head(self, model_cfg: Dict, in_channels: int) -> Optional[OrientedStandardRoIHead]:
        if model_cfg.get("RPN_ONLY", False):
            return None
        roi_extractor_cfg = dict(model_cfg.get("ROI_EXTRACTOR", {}))
        roi_extractor_cfg["OUT_CHANNELS"] = int(in_channels)
        roi_extractor_cfg.setdefault("FEATURE_KEY", self.roi_feature_key)
        roi_extractor_cfg.setdefault("POOL_SIZE", model_cfg.get("ROI_BOX_HEAD", {}).get("POOLER_RESOLUTION", 7))
        roi_extractor_cfg.setdefault("ANGLE_VERSION", model_cfg.get("ANGLE_VERSION", "le90"))
        roi_extractor_cfg.setdefault("FALLBACK_TO_HBB", True)
        roi_extractor_cfg.setdefault("SPATIAL_SCALE", 1.0 / float(self.feature_stride))

        bbox_roi_extractor = RotatedROIExtractor.from_config(roi_extractor_cfg)
        bbox_head = RotatedShared2FCBBoxHead(
            in_channels=int(in_channels),
            fc_out_channels=int(model_cfg.get("ROI_BOX_HEAD", {}).get("MLP_HEAD_DIM", 512)),
            roi_feat_size=int(roi_extractor_cfg["POOL_SIZE"]),
            num_classes=int(model_cfg.get("ROI_BOX_HEAD", {}).get("NUM_CLASSES", self.num_classes)),
            reg_class_agnostic=True,
            bbox_coder=DeltaXYWHAOBBoxCoder(
                angle_range=model_cfg.get("ANGLE_VERSION", "le90"),
                edge_swap=True,
                proj_xy=True,
            ),
            max_per_img=int(model_cfg.get("ROI_HEADS", {}).get("DETECTIONS_PER_IMG", 100)),
        )
        return OrientedStandardRoIHead(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            fg_iou_thresh=float(model_cfg.get("ROI_HEADS", {}).get("FG_IOU_THRESHOLD", 0.5)),
            bg_iou_thresh=float(model_cfg.get("ROI_HEADS", {}).get("BG_IOU_THRESHOLD", 0.5)),
            batch_size_per_image=int(model_cfg.get("ROI_HEADS", {}).get("BATCH_SIZE_PER_IMAGE", 256)),
            positive_fraction=float(model_cfg.get("ROI_HEADS", {}).get("POSITIVE_FRACTION", 0.25)),
        )

    def _infer_feature_strides(self, backbone_strides: List[int], neck_cfg: Dict) -> List[int]:
        if self.neck is None:
            return backbone_strides
        num_outs = int(neck_cfg.get("NUM_OUTS", len(backbone_strides)))
        strides = list(backbone_strides)
        while len(strides) < num_outs:
            strides.append(strides[-1] * 2)
        return strides[:num_outs]

    def _align_proposals_to_gt(self, proposal: BoxList, target: BoxList) -> BoxList:
        if len(proposal) == 0:
            matched = BoxList(proposal.bbox.clone(), proposal.size, proposal.mode)
            matched.add_field("labels", torch.zeros((0,), dtype=torch.long, device=proposal.bbox.device))
            matched.add_field("pair_labels", torch.zeros((0, 0), dtype=torch.long, device=proposal.bbox.device))
            matched.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long, device=proposal.bbox.device))
            return matched

        gt_boxes = target.bbox.float().to(proposal.bbox.device)
        gt_labels = target.get_field("labels").long().to(proposal.bbox.device)
        iou = self._pairwise_iou(proposal.bbox.float(), gt_boxes, target.size)
        best_iou, best_idx = iou.max(dim=1)
        matched_boxes = gt_boxes[best_idx]
        matched = BoxList(matched_boxes, target.size, target.mode)
        matched_labels = gt_labels[best_idx]
        matched_labels = torch.where(best_iou >= 0.3, matched_labels, torch.zeros_like(matched_labels))
        matched.add_field("labels", matched_labels)
        matched.add_field("matched_gt_idx", best_idx)
        matched.add_field("matched_iou", best_iou)
        matched.add_field("gt_relation_triplets", target.get_field("relation_triplets").long().to(proposal.bbox.device))
        return matched

    def _seed_relation_targets_from_matches(self, proposal: BoxList, matched: BoxList) -> None:
        pair_labels = torch.zeros((len(proposal), len(proposal)), dtype=torch.long, device=proposal.bbox.device)
        rel_triplets = []
        if len(proposal) > 0 and matched.has_field("gt_relation_triplets"):
            gt_rel = matched.get_field("gt_relation_triplets")
            matched_idx = matched.get_field("matched_gt_idx")
            matched_iou = matched.get_field("matched_iou")
            valid = matched_iou >= 0.3
            gt_to_pred: Dict[int, List[int]] = {}
            for pred_idx, gt_idx in enumerate(matched_idx.tolist()):
                if valid[pred_idx]:
                    gt_to_pred.setdefault(int(gt_idx), []).append(pred_idx)
            for subj, obj, pred in gt_rel.tolist():
                subj_preds = gt_to_pred.get(int(subj), [])
                obj_preds = gt_to_pred.get(int(obj), [])
                for subj_pred in subj_preds:
                    for obj_pred in obj_preds:
                        if subj_pred == obj_pred:
                            continue
                        pair_labels[subj_pred, obj_pred] = int(pred)
                        rel_triplets.append([subj_pred, obj_pred, int(pred)])
        proposal.add_field("pair_labels", pair_labels)
        proposal.add_field(
            "relation_triplets",
            torch.tensor(rel_triplets, dtype=torch.long, device=proposal.bbox.device)
            if rel_triplets
            else torch.zeros((0, 3), dtype=torch.long, device=proposal.bbox.device),
        )

    def _attach_relation_predictions(
        self,
        proposals: Sequence[BoxList],
        relation_predictions: Sequence[Dict[str, torch.Tensor]],
    ) -> List[BoxList]:
        outputs: List[BoxList] = []
        for proposal, rel_pred in zip(proposals, relation_predictions):
            pred = self._copy_boxlist(proposal)
            rel_logits = rel_pred["rel_logits"]
            pred.add_field("rel_pair_idx", rel_pred["pair_idx"])
            pred.add_field("rel_logits", rel_logits)
            pred.add_field(
                "pred_rel_scores",
                rel_logits.softmax(-1) if rel_logits.numel() > 0 else rel_logits.new_zeros((0, self.num_predicates)),
            )
            pred.add_field("subgraph_assign", rel_pred["subgraph_assign"])
            outputs.append(pred)
        return outputs

    def _refine_boxes(self, boxes: torch.Tensor, deltas: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if self.box_mode == "obb":
            width, height = float(size[0]), float(size[1])
            refined = boxes.clone()
            deltas = torch.tanh(deltas)
            refined[:, 0] = (refined[:, 0] + deltas[:, 0] * refined[:, 2] * 0.25).clamp(min=0.0, max=width)
            refined[:, 1] = (refined[:, 1] + deltas[:, 1] * refined[:, 3] * 0.25).clamp(min=0.0, max=height)
            refined[:, 2] = (refined[:, 2] * (1.0 + deltas[:, 2] * 0.25)).clamp(min=1e-3)
            refined[:, 3] = (refined[:, 3] * (1.0 + deltas[:, 3] * 0.25)).clamp(min=1e-3)
            if deltas.size(1) > 4:
                refined[:, 4] = refined[:, 4] + deltas[:, 4] * 15.0
            return refined
        width, height = float(size[0]), float(size[1])
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        deltas = torch.tanh(deltas)
        cx = cx + deltas[:, 0] * w * 0.25
        cy = cy + deltas[:, 1] * h * 0.25
        w = w * (1.0 + deltas[:, 2] * 0.25).clamp(min=0.5)
        h = h * (1.0 + deltas[:, 3] * 0.25).clamp(min=0.5)
        refined = torch.stack(
            [
                (cx - 0.5 * w).clamp(min=0.0, max=width),
                (cy - 0.5 * h).clamp(min=0.0, max=height),
                (cx + 0.5 * w).clamp(min=0.0, max=width),
                (cy + 0.5 * h).clamp(min=0.0, max=height),
            ],
            dim=1,
        )
        return refined

    def _pairwise_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
        if self.box_mode == "obb":
            boxlist1 = BoxList(boxes1, image_size, "xywha")
            boxlist2 = BoxList(boxes2, image_size, "xywha")
            return boxlist_iou(boxlist1, boxlist2, mode="auto")
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
        area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
        union = area1[:, None] + area2[None, :] - inter + 1e-6
        return inter / union

    def _decode_object_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.numel() == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=logits.device),
                torch.zeros((0,), dtype=torch.float32, device=logits.device),
            )
        if logits.size(-1) <= 1:
            return (
                torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device),
                torch.ones((logits.size(0),), dtype=torch.float32, device=logits.device),
            )
        probs = logits.softmax(-1)
        scores, labels = probs[:, 1:].max(dim=-1)
        return labels.long() + 1, scores.float()

    def _copy_boxlist(self, boxlist: BoxList) -> BoxList:
        copied = BoxList(boxlist.bbox.clone(), boxlist.size, boxlist.mode)
        for field in boxlist.fields():
            value = boxlist.get_field(field)
            if torch.is_tensor(value):
                value = value.clone()
            copied.add_field(field, value)
        return copied

    def _boxlist_mode(self) -> str:
        return "xyxy" if self.box_mode == "hbb" else "xywha"
