from __future__ import annotations

import torch

from .boxlist import BoxList as _BaseBoxList
from .boxlist_ops import obb_to_hbb


class BoxList(_BaseBoxList):
    """
    Compatibility wrapper around ``sgg.structures.boxlist.BoxList``.

    Older code in this project imports ``sgg.structures.boxes.BoxList`` and
    expects helpers such as ``boxes``, ``areas()``, ``centers()``, and
    ``as_hbb()``.
    """

    @property
    def boxes(self):
        return self.bbox

    @property
    def image_size(self):
        return self.size

    def __getitem__(self, item):
        bbox = self.bbox[item]
        if bbox.ndim == 1:
            bbox = bbox.unsqueeze(0)
        box = BoxList(bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if torch.is_tensor(v):
                if v.ndim >= 2 and v.shape[0] == len(self) and v.shape[1] == len(self):
                    box.add_field(k, v[item][:, item])
                elif v.shape[0] == len(self):
                    box.add_field(k, v[item])
                else:
                    box.add_field(k, v)
            elif isinstance(v, list):
                if isinstance(item, slice):
                    box.add_field(k, v[item])
                elif torch.is_tensor(item):
                    indices = item.tolist()
                    box.add_field(k, [v[idx] for idx in indices])
                elif isinstance(item, (list, tuple)):
                    box.add_field(k, [v[idx] for idx in item])
                else:
                    box.add_field(k, v[item])
            else:
                box.add_field(k, v)
        return box

    def to(self, device):
        bbox = self.bbox.to(device)
        box = BoxList(bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            box.add_field(k, v)
        return box

    def convert(self, mode):
        converted = super().convert(mode)
        box = BoxList(converted.bbox, converted.size, converted.mode)
        for k, v in converted.extra_fields.items():
            box.add_field(k, v)
        return box

    def areas(self):
        return self.area()

    def centers(self):
        if self.mode == "xyxy":
            x1, y1, x2, y2 = self.bbox.unbind(dim=1)
            return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dim=1)
        if self.mode == "xywh":
            x, y, w, h = self.bbox.unbind(dim=1)
            return torch.stack([x + w * 0.5, y + h * 0.5], dim=1)
        if self.mode == "xywha":
            return self.bbox[:, :2]
        raise ValueError(f"Unsupported mode: {self.mode}")

    def as_hbb(self):
        if self.mode == "xyxy":
            return self.bbox
        if self.mode == "xywh":
            return self.convert("xyxy").bbox
        if self.mode == "xywha":
            return obb_to_hbb(self).bbox
        raise ValueError(f"Unsupported mode: {self.mode}")
