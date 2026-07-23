from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from sgg.data.sample import DatasetMetadata, SGGSample
from sgg.data.transforms.common import NormalizeTransform, ResizeTransform
from sgg.structures.boxes import BoxList


class BaseSGGDataset(Dataset, ABC):
    def __init__(
        self,
        transforms=None,
        metadata: Optional[DatasetMetadata] = None,
        sgdet_detector_original_resolution: bool = False,
    ):
        self.transforms = transforms
        self.metadata = metadata
        self.sgdet_detector_original_resolution = bool(sgdet_detector_original_resolution)

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_item_dict(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, index: int):
        sample = self.get_item_dict(index)
        if isinstance(sample, SGGSample):
            image = sample.image
            target = sample.target
            meta = sample.meta
        else:
            image = sample["image"]
            target = sample["target"]
            meta = sample.get("meta", {})
        # Original STAR Sgdets uses two image streams: its patch detector sees
        # the original-resolution image, while the relation head consumes a
        # separately resized image/target pair (``sgd_data`` in RPCM).  Keep
        # the auxiliary stream in meta so the established three-value collate
        # contract and every non-sgdet dataset remain unchanged.
        if self.sgdet_detector_original_resolution:
            detector_image = image
            detector_target = target
            if self.transforms is not None:
                image, target = self.transforms(image, target)
                detector_image, detector_target = self._apply_detector_view_transforms(
                    detector_image,
                    detector_target,
                )
            meta = dict(meta)
            meta["sgdet_detector_image"] = detector_image
            # Detection only needs its original image extent; carrying a
            # second BoxList (and its potentially N×N relation fields) through
            # DataLoader metadata would duplicate substantial CPU/shared-memory
            # traffic for no benefit.
            meta["sgdet_detector_size"] = tuple(detector_target.size)
        elif self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target, meta

    def _apply_detector_view_transforms(self, image: torch.Tensor, target: BoxList):
        """Apply only non-geometric normalization to the raw detector view.

        Resize belongs solely to the relation stream.  Random geometry would
        need a shared sampled transform for both streams; reject it here rather
        than silently creating detector/target coordinate drift.  The STAR
        sgdet compatibility config therefore keeps augmentation disabled, as
        required by the frozen original detector route.
        """
        transforms = getattr(self.transforms, "transforms", None)
        if transforms is None:
            raise TypeError(
                "SGDET_DETECTOR_ORIGINAL_RESOLUTION requires a Compose transform pipeline."
            )
        for transform in transforms:
            if isinstance(transform, ResizeTransform):
                continue
            if isinstance(transform, NormalizeTransform):
                image, target = transform(image, target)
                continue
            raise ValueError(
                "SGDET_DETECTOR_ORIGINAL_RESOLUTION is incompatible with random geometric "
                f"transform {type(transform).__name__}; set DATASETS.*.AUGMENT=False or "
                "provide a shared dual-view transform."
            )
        return image, target
