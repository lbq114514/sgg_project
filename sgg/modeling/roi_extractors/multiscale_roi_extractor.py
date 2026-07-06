from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from sgg.modeling.core.obb_ops import convert_obb_angle_unit, get_boxlist_angle_unit
from sgg.modeling.layers.roi_align import MultiScaleROIAlign
from sgg.structures.boxlist import BoxList


class MultiScaleROIExtractor(nn.Module):
    """
    Multi-scale ROI extractor for FPN features.

    Use ``mode="hbb"`` for horizontal boxes and ``mode="obb"`` for rotated boxes.
    """

    def __init__(
        self,
        pool_size: int = 7,
        out_channels: int = 256,
        sampling_ratio: int = 0,
        aligned: bool = True,
        clockwise: bool = False,
        mode: str = "hbb",
        featmap_names: Optional[List[str]] = None,
        featmap_strides: Optional[List[int]] = None,
        canonical_scale: float = 224.0,
        canonical_level: int = 4,
        min_level: int = 2,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.out_channels = out_channels
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.clockwise = clockwise
        self.mode = mode
        self.featmap_names = featmap_names or ["p2", "p3", "p4", "p5"]
        self.featmap_strides = featmap_strides or [4, 8, 16, 32]
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.min_level = min_level

        self.pooler = MultiScaleROIAlign(
            output_size=self.pool_size,
            sampling_ratio=self.sampling_ratio,
            featmap_strides=self.featmap_strides,
            mode=self.mode,
            aligned=self.aligned,
            clockwise=self.clockwise,
        )
        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def from_config(cls, cfg):
        return cls(
            pool_size=cfg.get("POOL_SIZE", 7),
            out_channels=cfg.get("OUT_CHANNELS", 256),
            sampling_ratio=cfg.get("SAMPLING_RATIO", 0),
            aligned=cfg.get("ALIGNED", True),
            clockwise=cfg.get("CLOCKWISE", False),
            mode=cfg.get("MODE", "hbb"),
            featmap_names=cfg.get("FEATMAP_NAMES", ["p2", "p3", "p4", "p5"]),
            featmap_strides=cfg.get("FEATMAP_STRIDES", [4, 8, 16, 32]),
            canonical_scale=cfg.get("CANONICAL_SCALE", 224.0),
            canonical_level=cfg.get("CANONICAL_LEVEL", 4),
            min_level=cfg.get("MIN_LEVEL", 2),
        )

    def _select_feature_maps(
        self, features: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]
    ) -> List[torch.Tensor]:
        if isinstance(features, torch.Tensor):
            return [features]
        if isinstance(features, (list, tuple)):
            return list(features)
        if isinstance(features, dict):
            selected = []
            for name in self.featmap_names:
                if name in features:
                    selected.append(features[name])
            if selected:
                return selected
            return [features[k] for k in sorted(features.keys())]
        raise TypeError(f"Unsupported features type: {type(features)}")

    def _convert_boxes_to_roi_format(self, proposals: List[BoxList]) -> torch.Tensor:
        concat_boxes = []
        device = None
        dtype = None
        box_dim = 4 if self.mode == "hbb" else 5

        for batch_idx, boxlist in enumerate(proposals):
            if boxlist.mode == "xyxy" and self.mode != "hbb":
                raise ValueError("MultiScaleROIExtractor with mode='obb' expects xywha boxes")
            if boxlist.mode == "xywha" and self.mode != "obb":
                raise ValueError("MultiScaleROIExtractor with mode='hbb' expects xyxy boxes")

            boxes = boxlist.bbox
            if boxes.numel() == 0:
                continue
            if boxlist.mode == "xywha":
                boxes = convert_obb_angle_unit(boxes, get_boxlist_angle_unit(boxlist), "radian")

            device = boxes.device
            dtype = boxes.dtype
            img_inds = torch.full(
                (boxes.shape[0], 1), batch_idx, dtype=boxes.dtype, device=boxes.device
            )
            concat_boxes.append(torch.cat([img_inds, boxes], dim=1))

        if not concat_boxes:
            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32
            return torch.zeros((0, box_dim + 1), dtype=dtype, device=device)

        return torch.cat(concat_boxes, dim=0)

    def _compute_levels(self, rois: torch.Tensor) -> torch.Tensor:
        if rois.numel() == 0:
            return rois.new_zeros((0,), dtype=torch.long)

        if self.mode == "hbb":
            widths = (rois[:, 3] - rois[:, 1]).clamp(min=1e-6)
            heights = (rois[:, 4] - rois[:, 2]).clamp(min=1e-6)
        else:
            widths = rois[:, 3].clamp(min=1e-6)
            heights = rois[:, 4].clamp(min=1e-6)

        scales = torch.sqrt(widths * heights)
        target_levels = torch.floor(
            self.canonical_level + torch.log2(scales / self.canonical_scale + 1e-6)
        )
        target_levels = target_levels.clamp(
            min=self.min_level,
            max=self.min_level + len(self.featmap_strides) - 1,
        )
        return (target_levels.to(torch.long) - self.min_level)

    def forward(
        self,
        features: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor],
        proposals: List[BoxList],
    ) -> torch.Tensor:
        feat_list = self._select_feature_maps(features)
        rois = self._convert_boxes_to_roi_format(proposals)
        levels = self._compute_levels(rois)
        pooled = self.pooler(feat_list, rois, levels)
        return self.post(pooled)
