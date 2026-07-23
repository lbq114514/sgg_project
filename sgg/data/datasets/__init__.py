from .registry import DATASETS
from .generic_sgg_json import GenericSGGJsonDataset
from .star import STARDataset
from .vg_like import VGLikeSGGDataset, OBBSGGJsonDataset

DATASET_CLS_MAP = DATASETS.as_dict()

__all__ = [
    "DATASETS",
    "DATASET_CLS_MAP",
    "GenericSGGJsonDataset",
    "STARDataset",
    "VGLikeSGGDataset",
    "OBBSGGJsonDataset",
]
