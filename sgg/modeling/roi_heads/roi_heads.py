from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn

from sgg.modeling.roi_heads.relation_head import build_roi_relation_head


def _cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


class CombinedROIHeads(nn.ModuleDict):
    """
    Minimal CombinedROIHeads implementation for this project.

    The predictor-specific branching from RPCM is intentionally excluded.
    """

    def __init__(self, cfg, heads):
        super().__init__(heads)
        self.cfg = cfg
        self.type = cfg.get("TYPE", "CV")
        self.relation_on = bool(_cfg_get(cfg, "MODEL", "RELATION_ON", default=True))
        self.attribute_on = bool(_cfg_get(cfg, "MODEL", "ATTRIBUTE_ON", default=False))
        self.mask_on = bool(_cfg_get(cfg, "MODEL", "MASK_ON", default=False))
        self.keypoint_on = bool(_cfg_get(cfg, "MODEL", "KEYPOINT_ON", default=False))

        use_gt_box = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_BOX", default=False))
        use_gt_object_label = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_OBJECT_LABEL", default=False)
        )
        if use_gt_box:
            self.mode = "predcls" if use_gt_object_label else "sgcls"
        else:
            self.mode = "sgdet"

    def forward(
        self,
        features,
        proposals,
        targets=None,
        logger=None,
        **kwargs,
    ):
        losses = {}
        detections = proposals
        x = features

        if self.type == "CV" and "box" in self:
            x, detections, loss_box = self.box(features, proposals, targets, logger=logger)
            if not self.relation_on:
                losses.update(loss_box)

        if self.attribute_on and "attribute" in self:
            x, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.mask_on and "mask" in self:
            x, detections, loss_mask = self.mask(features, detections, targets)
            losses.update(loss_mask)

        if self.keypoint_on and "keypoint" in self:
            x, detections, loss_keypoint = self.keypoint(features, detections, targets)
            losses.update(loss_keypoint)

        if self.relation_on and "relation" in self:
            x, detections, loss_relation = self.relation(features, detections, targets, logger=logger, **kwargs)
            losses.update(loss_relation)

        return x, detections, losses


def build_roi_heads(cfg, in_channels, RS=None, RS_conf=None):
    del RS, RS_conf
    if _cfg_get(cfg, "MODEL", "RETINANET_ON", default=False):
        return []

    heads = []
    model_cfg = cfg.get("MODEL", {})

    if _cfg_get(cfg, "MODEL", "RELATION_ON", default=True):
        heads.append(
            (
                "relation",
                build_roi_relation_head(cfg, int(in_channels)),
            )
        )

    if not heads:
        return []

    return CombinedROIHeads(cfg, OrderedDict(heads))
