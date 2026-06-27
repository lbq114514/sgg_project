from typing import Dict, List, Union

import torch
import torch.nn as nn

from sgg.modeling.layers.roi_align import ROIAlign
from sgg.structures.boxlist import BoxList


class HBBROIExtractor(nn.Module):
    """Single-scale ROI extractor for horizontal boxes."""

    def __init__(
        self,
        pool_size: int = 7,
        out_channels: int = 256,
        sampling_ratio: int = 0,
        aligned: bool = True,
        spatial_scale: float = 1.0,
        feature_key: str = "p2",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.out_channels = out_channels
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.spatial_scale = spatial_scale
        self.feature_key = feature_key

        self.pooler = ROIAlign(
            output_size=self.pool_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
            mode="hbb",
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
            spatial_scale=cfg.get("SPATIAL_SCALE", 1.0),
            feature_key=cfg.get("FEATURE_KEY", "p2"),
        )

    def _select_feature_map(
        self, features: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if isinstance(features, (list, tuple)):
            if len(features) == 0:
                raise ValueError("features list is empty")
            return features[0]
        if isinstance(features, dict):
            if not features:
                raise ValueError("features dict is empty")
            if self.feature_key in features:
                return features[self.feature_key]
            first_key = sorted(features.keys())[0]
            return features[first_key]
        raise TypeError(f"Unsupported features type: {type(features)}")

    def _to_hbb_proposals(self, proposals: List[BoxList]) -> List[BoxList]:
        converted = []
        for boxlist in proposals:
            if boxlist.mode == "xyxy":
                converted.append(boxlist)
            else:
                raise ValueError(
                    f"HBBROIExtractor expects 'xyxy' BoxList, got {boxlist.mode}"
                )
        return converted

    def forward(
        self,
        features: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor],
        proposals: List[BoxList],
    ) -> torch.Tensor:
        feat = self._select_feature_map(features)
        pooled = self.pooler(feat, self._to_hbb_proposals(proposals))
        return self.post(pooled)
