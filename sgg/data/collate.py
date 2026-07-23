from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from sgg.structures.boxes import BoxList


def pad_images(images: List[torch.Tensor], size_divisible: int = 0) -> torch.Tensor:
    max_h = max(im.shape[1] for im in images)
    max_w = max(im.shape[2] for im in images)

    if size_divisible > 0:
        max_h = ((max_h + size_divisible - 1) // size_divisible) * size_divisible
        max_w = ((max_w + size_divisible - 1) // size_divisible) * size_divisible

    padded = []
    for im in images:
        pad_h = max_h - im.shape[1]
        pad_w = max_w - im.shape[2]
        padded.append(torch.nn.functional.pad(im, (0, pad_w, 0, pad_h)))
    return torch.stack(padded, dim=0)


def sgg_collate_fn(
    batch: List[Tuple[torch.Tensor, BoxList, Dict]],
    size_divisible: int = 0,
):
    images, targets, metas = zip(*batch)
    return pad_images(list(images), size_divisible=size_divisible), list(targets), list(metas)
