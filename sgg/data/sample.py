from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from sgg.structures.boxes import BoxList


@dataclass
class SGGSample:
    image: torch.Tensor
    target: BoxList
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    dataset_name: str
    box_mode: str
    num_classes: int
    num_predicates: int
    categories: Optional[Any] = None
    predicates: Optional[Any] = None
