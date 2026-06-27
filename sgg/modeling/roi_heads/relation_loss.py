from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


class RelationLossEvaluator:
    def __init__(self, cfg: dict):
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.num_rel_classes = int(cfg["MODEL"]["ROI_RELATION_HEAD"]["NUM_CLASSES"])
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.loss_type = str(rel_cfg.get("PREDICATE_LOSS_TYPE", "ce")).lower()
        self.cb_beta = float(rel_cfg.get("PREDICATE_CLASS_BALANCED_BETA", 0.999))
        self.bg_loss_weight = float(rel_cfg.get("PREDICATE_BG_LOSS_WEIGHT", 1.0))
        self.logit_adjust_tau = float(rel_cfg.get("PREDICATE_LOGIT_ADJUST_TAU", 1.0))
        counts = rel_cfg.get("PREDICATE_COUNTS", [])
        self.predicate_counts = [int(v) for v in counts] if counts else []
        self.obj_loss_weight = float(rel_cfg.get("OBJECT_REFINE_LOSS_WEIGHT", 1.0))

    def _counts_tensor(self, device: torch.device) -> torch.Tensor:
        counts = torch.ones((self.num_rel_classes,), dtype=torch.float32, device=device)
        if self.predicate_counts:
            n = min(len(self.predicate_counts), self.num_rel_classes)
            counts[:n] = torch.as_tensor(self.predicate_counts[:n], dtype=torch.float32, device=device)
        counts = counts.clamp(min=1.0)
        return counts

    def _class_balanced_weight(self, device: torch.device) -> torch.Tensor:
        counts = self._counts_tensor(device)
        if self.cb_beta <= 0.0 or self.cb_beta >= 1.0:
            weights = 1.0 / counts
        else:
            beta = torch.tensor(self.cb_beta, dtype=torch.float32, device=device)
            weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / weights[1:].mean().clamp(min=1e-6) if weights.numel() > 1 else weights
        weights[0] = self.bg_loss_weight
        return weights

    def _apply_logit_adjustment(self, logits: torch.Tensor) -> torch.Tensor:
        if self.loss_type != "logit_adjusted" or self.logit_adjust_tau == 0.0:
            return logits
        counts = self._counts_tensor(logits.device)
        prior = counts / counts.sum().clamp(min=1.0)
        return logits + self.logit_adjust_tau * prior.clamp(min=1e-12).log()

    def __call__(self, proposals: Sequence, rel_labels, relation_logits, refine_logits=None, cls_new=None):
        del cls_new
        device = None
        if isinstance(relation_logits, torch.Tensor):
            device = relation_logits.device
        elif relation_logits:
            device = relation_logits[0].device
        else:
            device = torch.device("cpu")

        if isinstance(relation_logits, list):
            relation_logits = torch.cat(relation_logits, dim=0) if relation_logits else torch.zeros((0, self.num_rel_classes), device=device)
        if isinstance(rel_labels, list):
            rel_labels = torch.cat(rel_labels, dim=0) if rel_labels else torch.zeros((0,), dtype=torch.long, device=device)

        if relation_logits.numel() == 0 or rel_labels.numel() == 0:
            loss_relation = torch.zeros((), device=device)
        else:
            labels = rel_labels.clamp(min=0, max=self.num_rel_classes - 1)
            relation_logits = self._apply_logit_adjustment(relation_logits)
            weight = self._class_balanced_weight(relation_logits.device) if self.loss_type == "class_balanced" else None
            loss_relation = F.cross_entropy(relation_logits, labels, weight=weight)

        loss_refine_obj = None
        if refine_logits is not None:
            if isinstance(refine_logits, list):
                refine_logits = (
                    torch.cat(refine_logits, dim=0)
                    if refine_logits
                    else torch.zeros((0, self.num_obj_classes), device=device)
                )
            if refine_logits.numel() > 0:
                fg_labels = torch.cat(
                    [
                        proposal.get_field("gt_labels") if proposal.has_field("gt_labels") else proposal.get_field("labels")
                        for proposal in proposals
                    ],
                    dim=0,
                )
                fg_labels = fg_labels.clamp(min=0, max=refine_logits.size(1) - 1)
                loss_refine_obj = F.cross_entropy(refine_logits, fg_labels) * self.obj_loss_weight

        return loss_relation, loss_refine_obj


def make_roi_relation_loss_evaluator(cfg: dict):
    return RelationLossEvaluator(cfg)
