from .hbb_roi_extractor import HBBROIExtractor


class SimpleROIExtractor(HBBROIExtractor):
    """Backward-compatible alias of the single-scale HBB ROI extractor."""

    def __init__(
        self,
        pool_size: int = 7,
        out_channels: int = 256,
        sampling_ratio: int = 0,
        aligned: bool = True,
        spatial_scale: float = 1.0,
        feature_key: str = "p2",
    ):
        super().__init__(
            pool_size=pool_size,
            out_channels=out_channels,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
            spatial_scale=spatial_scale,
            feature_key=feature_key,
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
