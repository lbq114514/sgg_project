import torch


class BoxList:
    """
    Unified bounding box container for HBB / OBB.

    Supported modes:
        xyxy  : horizontal box (x1, y1, x2, y2)
        xywh  : horizontal box (x, y, w, h)
        xywha : oriented box   (cx, cy, w, h, angle)

    Args:
        bbox (Tensor):
            Shape (N, 4) or (N, 5)

        image_size (tuple):
            (width, height)

        mode (str):
            Box mode.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        if mode not in ("xyxy", "xywh", "xywha"):
            raise ValueError(f"Unsupported mode: {mode}")

        if bbox.ndim != 2:
            raise ValueError("bbox must be Nx4 or Nx5")

        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    # ---------------------------------------------------------
    # extra fields
    # ---------------------------------------------------------
    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    # ---------------------------------------------------------
    # basic
    # ---------------------------------------------------------
    def __len__(self):
        return self.bbox.shape[0]

    def __repr__(self):
        return (
            f"BoxList(num_boxes={len(self)}, "
            f"image_width={self.size[0]}, "
            f"image_height={self.size[1]}, "
            f"mode={self.mode})"
        )

    def __getitem__(self, item):
        bbox = self.bbox[item]
        if bbox.ndim == 1:
            bbox = bbox.unsqueeze(0)
        box = BoxList(bbox, self.size, self.mode)

        for k, v in self.extra_fields.items():
            sliced = v
            if torch.is_tensor(v):
                if isinstance(item, torch.Tensor):
                    index = item.long().to(v.device)
                else:
                    index = item
                if v.ndim >= 2 and v.shape[0] == len(self) and v.shape[1] == len(self):
                    sliced = v[index][:, index]
                elif v.ndim >= 1 and v.shape[0] == len(self):
                    sliced = v[index]
            elif isinstance(v, list):
                if isinstance(item, torch.Tensor):
                    sliced = [v[int(i)] for i in item.detach().cpu().tolist()]
                elif isinstance(item, slice):
                    sliced = v[item]
                else:
                    sliced = v[int(item)]
            box.add_field(k, sliced)

        return box

    def to(self, device):
        bbox = self.bbox.to(device)
        box = BoxList(bbox, self.size, self.mode)

        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            box.add_field(k, v)

        return box

    # ---------------------------------------------------------
    # convert
    # ---------------------------------------------------------
    def convert(self, mode):
        """
        Only supports HBB conversion internally.

        OBB <-> HBB conversion should use boxlist_ops.py
        """
        if mode == self.mode:
            return self

        if self.mode == "xyxy" and mode == "xywh":
            x1, y1, x2, y2 = self.bbox.unbind(dim=1)
            bbox = torch.stack(
                [x1, y1, x2 - x1, y2 - y1],
                dim=1
            )

        elif self.mode == "xywh" and mode == "xyxy":
            x, y, w, h = self.bbox.unbind(dim=1)
            bbox = torch.stack(
                [x, y, x + w, y + h],
                dim=1
            )

        else:
            raise NotImplementedError(
                "Use boxlist_ops for OBB conversion"
            )

        box = BoxList(bbox, self.size, mode)

        for k, v in self.extra_fields.items():
            box.add_field(k, v)

        return box

    # ---------------------------------------------------------
    # geometry
    # ---------------------------------------------------------
    def area(self):
        if self.mode == "xyxy":
            x1, y1, x2, y2 = self.bbox.unbind(dim=1)
            return (x2 - x1).clamp(min=0) * \
                   (y2 - y1).clamp(min=0)

        elif self.mode == "xywh":
            _, _, w, h = self.bbox.unbind(dim=1)
            return w.clamp(min=0) * h.clamp(min=0)

        elif self.mode == "xywha":
            _, _, w, h, _ = self.bbox.unbind(dim=1)
            return w.clamp(min=0) * h.clamp(min=0)

        else:
            raise NotImplementedError

    def clip_to_image(self, remove_empty=True):
        """
        Clip HBB to image boundary.

        OBB is returned unchanged.
        """
        if self.mode == "xywha":
            return self

        box = self.convert("xyxy")
        w, h = self.size

        box.bbox[:, 0].clamp_(min=0, max=w)
        box.bbox[:, 2].clamp_(min=0, max=w)
        box.bbox[:, 1].clamp_(min=0, max=h)
        box.bbox[:, 3].clamp_(min=0, max=h)

        if remove_empty:
            keep = (
                (box.bbox[:, 2] > box.bbox[:, 0]) &
                (box.bbox[:, 3] > box.bbox[:, 1])
            )
            return box[keep]

        return box
