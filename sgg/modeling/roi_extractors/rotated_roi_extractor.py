from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from sgg.modeling.core.obb_ops import convert_obb_angle_unit, get_boxlist_angle_unit
from sgg.modeling.layers.roi_align import MultiScaleROIAlign
from sgg.structures.boxlist import BoxList
from sgg.structures.boxlist_ops import obb_to_hbb


class RotatedROIExtractor(nn.Module):
    """
    ROI feature extractor for OBB proposals.

    Preferred path:
        - use ``ROIAlign(mode="obb")`` backed by ``mmcv.ops.RoIAlignRotated``

    Fallback path:
        - convert OBB proposals to HBB envelopes
        - use standard ROIAlign on the converted boxes

    This keeps OBB-aware behavior when mmcv is available, while still
    producing proposal-conditioned features in dependency-light setups.
    """

    def __init__(
        self,
        pool_size: int = 7,
        out_channels: int = 256,
        sampling_ratio: int = 0,
        aligned: bool = True,
        clockwise: bool = False,
        angle_version: str = "oc",
        fallback_to_hbb: bool = True,
        spatial_scale: float = 1.0,
        feature_key: str = "p2",
        featmap_names: Optional[List[str]] = None,
        featmap_strides: Optional[List[int]] = None,
        finest_scale: float = 56.0,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.out_channels = out_channels
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.clockwise = clockwise
        self.angle_version = angle_version
        self.fallback_to_hbb = fallback_to_hbb
        self.spatial_scale = spatial_scale
        self.feature_key = feature_key
        self.featmap_names = featmap_names or ["p2", "p3", "p4", "p5"]
        self.featmap_strides = featmap_strides or [4, 8, 16, 32]
        self.finest_scale = finest_scale

        self.obb_pooler = None
        self.hbb_pooler = MultiScaleROIAlign(
            output_size=self.pool_size,
            sampling_ratio=self.sampling_ratio,
            featmap_strides=self.featmap_strides,
            mode="hbb",
            aligned=self.aligned,
            clockwise=self.clockwise,
        )
        try:
            self.obb_pooler = MultiScaleROIAlign(
                output_size=self.pool_size,
                sampling_ratio=self.sampling_ratio,
                featmap_strides=self.featmap_strides,
                mode="obb",
                aligned=self.aligned,
                clockwise=self.clockwise,
            )
        except ImportError:
            self.obb_pooler = None

    @classmethod
    def from_config(cls, cfg):
        return cls(
            pool_size=cfg.get("POOL_SIZE", 7),
            out_channels=cfg.get("OUT_CHANNELS", 256),
            sampling_ratio=cfg.get("SAMPLING_RATIO", 0),
            aligned=cfg.get("ALIGNED", True),
            clockwise=cfg.get("CLOCKWISE", False),
            angle_version=cfg.get("ANGLE_VERSION", "oc"),
            fallback_to_hbb=cfg.get("FALLBACK_TO_HBB", True),
            spatial_scale=cfg.get("SPATIAL_SCALE", 1.0),
            feature_key=cfg.get("FEATURE_KEY", "p2"),
            featmap_names=cfg.get("FEATMAP_NAMES", ["p2", "p3", "p4", "p5"]),
            featmap_strides=cfg.get("FEATMAP_STRIDES", [4, 8, 16, 32]),
            finest_scale=cfg.get("FINEST_SCALE", 56.0),
        )

    def _select_feature_maps(
        self, features: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]
    ) -> List[torch.Tensor]:
        if isinstance(features, torch.Tensor):
            return [features]

        if isinstance(features, (list, tuple)):
            if len(features) == 0:
                raise ValueError("features list is empty")
            return list(features[: len(self.featmap_strides)])

        if isinstance(features, dict):
            if not features:
                raise ValueError("features dict is empty")
            selected = [features[name] for name in self.featmap_names if name in features]
            if selected:
                return selected
            if self.feature_key in features:
                return [features[self.feature_key]]
            return [features[k] for k in sorted(features.keys())[: len(self.featmap_strides)]]

        raise TypeError(f"Unsupported features type: {type(features)}")

    def _convert_boxes_to_roi_format(self, proposals: List[BoxList]) -> torch.Tensor:
        concat_boxes = []
        device = None
        dtype = None
        for batch_idx, boxlist in enumerate(proposals):
            boxes = boxlist.bbox
            if boxes.numel() == 0:
                continue
            if boxlist.mode == "xywha":
                boxes = convert_obb_angle_unit(boxes, get_boxlist_angle_unit(boxlist), "radian")
            device = boxes.device
            dtype = boxes.dtype
            img_inds = torch.full((boxes.shape[0], 1), batch_idx, dtype=dtype, device=device)
            concat_boxes.append(torch.cat([img_inds, boxes], dim=1))
        if not concat_boxes:
            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32
            return torch.zeros((0, 6), dtype=dtype, device=device)
        return torch.cat(concat_boxes, dim=0)

    def _compute_levels(self, rois: torch.Tensor) -> torch.Tensor:
        if rois.numel() == 0:
            return rois.new_zeros((0,), dtype=torch.long)
        widths = rois[:, 3].clamp(min=1e-6)
        heights = rois[:, 4].clamp(min=1e-6)
        scales = torch.sqrt(widths * heights)
        target_levels = torch.floor(torch.log2(scales / self.finest_scale + 1e-6))
        return target_levels.clamp(min=0, max=len(self.featmap_strides) - 1).to(torch.long)

    def _to_hbb_proposals(self, proposals: List[BoxList]) -> List[BoxList]:
        hbb_proposals = []
        for boxlist in proposals:
            if boxlist.mode == "xywha":
                hbb_proposals.append(self._obb_boxlist_to_hbb(boxlist))
            elif boxlist.mode == "xyxy":
                hbb_proposals.append(boxlist)
            else:
                raise ValueError(f"Unsupported BoxList mode for ROI extraction: {boxlist.mode}")
        return hbb_proposals

    def _obb_boxlist_to_hbb(self, boxlist: BoxList) -> BoxList:
        return obb_to_hbb(boxlist, version=self.angle_version)

    def forward(
        self,
        features: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor],
        proposals: List[BoxList],
    ) -> torch.Tensor:
        feat_list = self._select_feature_maps(features)
        if len(feat_list) != len(self.featmap_strides):
            raise ValueError(
                f"Expected {len(self.featmap_strides)} feature maps for strides "
                f"{self.featmap_strides}, but got {len(feat_list)}"
            )
        rois = self._convert_boxes_to_roi_format(proposals)
        levels = self._compute_levels(rois)

        if self.obb_pooler is not None:
            return self.obb_pooler(feat_list, rois, levels)

        if not self.fallback_to_hbb:
            raise ImportError(
                "mmcv.ops.RoIAlignRotated is not available and FALLBACK_TO_HBB is disabled."
            )

        hbb_rois = self._convert_boxes_to_roi_format(self._to_hbb_proposals(proposals))
        return self.hbb_pooler(feat_list, hbb_rois, levels)
