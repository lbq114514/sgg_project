from __future__ import annotations

import torch
import torch.nn as nn

from sgg.modeling.box_coder.box_coder import DeltaXYWHAOBBoxCoder
from sgg.modeling.heads.rotated_bbox_head import RotatedBBoxHead


class RotatedShared2FCBBoxHead(RotatedBBoxHead):
    """
    Two-FC rotated bbox head.

    This mirrors the practical shape of mmrotate's Shared2FC head while using
    plain PyTorch modules and the project's lightweight bbox-head base class.
    """

    def __init__(
        self,
        in_channels: int,
        fc_out_channels: int = 1024,
        roi_feat_size: int = 7,
        num_classes: int = 1,
        reg_class_agnostic: bool = True,
        bbox_coder: object | None = None,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_per_img: int = 100,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 1.0,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            roi_feat_size=roi_feat_size,
            reg_class_agnostic=reg_class_agnostic,
            bbox_coder=bbox_coder or DeltaXYWHAOBBoxCoder(),
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            max_per_img=max_per_img,
            loss_cls_weight=loss_cls_weight,
            loss_bbox_weight=loss_bbox_weight,
        )
        self.fc_out_channels = int(fc_out_channels)
        input_dim = self.in_channels * self.roi_feat_area

        self.shared_fcs = nn.ModuleList(
            [
                nn.Linear(input_dim, self.fc_out_channels),
                nn.Linear(self.fc_out_channels, self.fc_out_channels),
            ]
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        reg_out_dim = 5 if self.reg_class_agnostic else 5 * self.num_classes
        self.fc_reg = nn.Linear(self.fc_out_channels, reg_out_dim)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.flatten(1)

        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
