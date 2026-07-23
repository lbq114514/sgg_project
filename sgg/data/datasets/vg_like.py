from __future__ import annotations

from .generic_sgg_json import GenericSGGJsonDataset
from .registry import DATASETS


@DATASETS.register("vg_like_sgg")
class VGLikeSGGDataset(GenericSGGJsonDataset):
    """VG-style alias of the generic JSON dataset."""


@DATASETS.register("obb_sgg_json")
class OBBSGGJsonDataset(GenericSGGJsonDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("box_mode", "obb")
        super().__init__(*args, **kwargs)
