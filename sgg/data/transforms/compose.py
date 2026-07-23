from __future__ import annotations

from typing import Iterable


class Compose:
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
