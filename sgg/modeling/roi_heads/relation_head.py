from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from sgg.modeling.roi_heads.ppg import PairProposalGenerator
from sgg.modeling.roi_heads.relation_inference import make_roi_relation_post_processor
from sgg.modeling.roi_heads.relation_loss import make_roi_relation_loss_evaluator
from sgg.modeling.roi_heads.relation_sampling import make_roi_relation_samp_processor
from sgg.modeling.roi_heads.sema_filter import SemanticPairFilter
from sgg.modeling.roi_heads.roi_relation_feature_extractors import (
    make_roi_box_feature_extractor,
    make_roi_relation_feature_extractor,
)
from sgg.modeling.roi_heads.roi_relation_predictors import make_roi_relation_predictor


def _relation_feature_dim(cfg: dict, default: int) -> int:
    return int(cfg["MODEL"].get("ROI_BOX_HEAD", {}).get("MLP_HEAD_DIM", default))


def _to_onehot_logits(labels: torch.Tensor, num_classes: int, fill: float = 1000.0) -> torch.Tensor:
    logits = labels.new_full((labels.numel(), num_classes), -fill, dtype=torch.float32)
    if labels.numel() > 0:
        row_idx = torch.arange(labels.numel(), device=labels.device)
        logits[row_idx, labels.long()] = fill
    return logits


class ROIRelationHead(nn.Module):
    """
    Generic relation head scaffold.

    This mirrors the reference architecture layout:
        sampling -> object/union feature extraction -> predictor ->
        postprocess/loss

    Predictor internals remain intentionally separate.
    """

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        feat_dim = _relation_feature_dim(cfg, self.box_feature_extractor.out_channels)
        self.obj_feature_dim = feat_dim
        self.local_box_feature_proj = (
            nn.Linear(self.box_feature_extractor.out_channels, feat_dim)
            if self.box_feature_extractor.out_channels != feat_dim
            else nn.Identity()
        )
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        self.ppg = PairProposalGenerator(cfg)
        self.sema_filter = SemanticPairFilter(cfg)
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.type = cfg.get("TYPE", "CV")
        self.use_union_box = bool(rel_cfg.get("PREDICT_USE_VISION", True))
        self.use_gt_box = bool(rel_cfg.get("USE_GT_BOX", False))
        self.use_gt_object_label = bool(rel_cfg.get("USE_GT_OBJECT_LABEL", False))
        self.predictor_name = str(rel_cfg.get("PREDICTOR", ""))
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])

    def _feature_device(self, features) -> torch.device:
        if isinstance(features, torch.Tensor):
            return features.device
        if isinstance(features, (list, tuple)):
            if not features:
                return torch.device("cpu")
            return features[0].device
        if isinstance(features, dict):
            if not features:
                return torch.device("cpu")
            first_key = next(iter(features))
            return features[first_key].device
        return torch.device("cpu")

    def _collect_refine_logits_from_proposals(self, proposals):
        refine_logits = []
        for proposal in proposals:
            if proposal.has_field("predict_logits"):
                refine_logits.append(proposal.get_field("predict_logits"))
            elif proposal.has_field("pred_logits"):
                refine_logits.append(proposal.get_field("pred_logits"))
        return refine_logits if refine_logits else None

    def forward(
        self,
        features,
        proposals: Sequence,
        targets=None,
        logger=None,
        OBj=None,
        s_f=None,
        **kwargs,
    ):
        del s_f, kwargs
        if self.training:
            with torch.no_grad():
                if self.use_gt_box:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(self._feature_device(features), proposals)
            for proposal, pair_idx in zip(proposals, rel_pair_idxs):
                proposal.add_field("base_rel_pair_idxs", pair_idx)
            filtered_pair_idxs = []
            for proposal, pair_idx in zip(proposals, rel_pair_idxs):
                if proposal.has_field("labels"):
                    pair_idx = self.sema_filter.filter_pairs(pair_idx, proposal.get_field("labels").long())
                if self.ppg.enabled and self.ppg.filter_method == "PPG":
                    filtered_pair_idx = self.ppg.filter_pairs(proposal, pair_idx)
                else:
                    filtered_pair_idx = pair_idx
                filtered_pair_idxs.append(filtered_pair_idx)
                proposal.add_field("pruned_rel_pair_idxs", filtered_pair_idx)
            rel_pair_idxs = filtered_pair_idxs

        if self.use_gt_box and self.use_gt_object_label and self.predictor_name == "RPCM":
            for proposal in proposals:
                labels = proposal.get_field("labels").long().clamp(min=0, max=self.num_obj_classes)
                predict_logits = _to_onehot_logits(labels, num_classes=self.num_obj_classes + 1)
                proposal.add_field("predict_logits", predict_logits.to(proposal.bbox.device))
                proposal.add_field("pred_scores", torch.ones(len(labels), device=proposal.bbox.device))
                proposal.add_field("pred_labels", labels.to(proposal.bbox.device))

        if OBj is not None and hasattr(OBj, "bbox_roi_extractor") and hasattr(OBj, "bbox_head"):
            roi_feats = OBj.bbox_roi_extractor(features, list(proposals))
            roi_features = OBj.bbox_head(roi_feats)
        else:
            roi_features = self.box_feature_extractor(features, proposals)
            roi_features = self.local_box_feature_proj(roi_features)

        union_features = (
            self.union_feature_extractor(features, proposals, rel_pair_idxs, OBj=OBj)
            if self.use_union_box
            else None
        )

        predictor_output = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_labels,
            rel_binarys,
            roi_features,
            union_features,
            logger,
        )
        if not isinstance(predictor_output, tuple):
            raise TypeError("Relation predictor must return a tuple.")
        if len(predictor_output) == 2:
            relation_logits, add_losses = predictor_output
            refine_logits = self._collect_refine_logits_from_proposals(proposals)
        elif len(predictor_output) == 3:
            relation_logits, refine_logits, add_losses = predictor_output
        else:
            raise ValueError(
                "Relation predictor must return (relation_logits, add_losses) "
                "or (relation_logits, refine_logits, add_losses)."
            )

        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}

        loss_relation, loss_refine_obj = self.loss_evaluator(
            proposals,
            rel_labels,
            relation_logits,
            refine_logits=refine_logits,
        )
        output_losses = {
            "loss_rel": loss_relation,
        }
        if loss_refine_obj is not None:
            output_losses["loss_refine_obj"] = loss_refine_obj
        output_losses.update(add_losses)
        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg: dict, in_channels: int):
    return ROIRelationHead(cfg, in_channels)
