import math
import torch


class AnchorGenerator:
    """
    Unified anchor generator for HBB / OBB.

    HBB mode outputs anchors in:
        (x1, y1, x2, y2)

    OBB mode outputs anchors in:
        (cx, cy, w, h, angle)

    Args:
        sizes (tuple[tuple[int]]):
            Anchor sizes for each feature level.
        aspect_ratios (tuple[float]):
            Aspect ratios used at each location.
        strides (tuple[int]):
            FPN strides.
        angles (tuple[float], optional):
            Used only in OBB mode.
        mode (str):
            "hbb" or "obb"
        offset (float):
            Center offset in [0, 1].  The mmdet/mmrotate AnchorGenerator
            used by the STAR detector defaults to 0.0.
    """

    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(0.5, 1.0, 2.0),
        strides=(4, 8, 16, 32, 64),
        angles=(0.0,),
        mode="hbb",
        offset=0.0,
    ):
        if mode not in ("hbb", "obb"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.angles = angles
        self.mode = mode
        self.offset = offset
        self.num_levels = len(strides)

        if len(self.sizes) != len(self.strides):
            raise ValueError("sizes and strides should have the same length")

    def num_anchors_per_location(self):
        if self.mode == "hbb":
            return [len(s) * len(self.aspect_ratios) for s in self.sizes]
        return [len(s) * len(self.aspect_ratios) * len(self.angles) for s in self.sizes]

    def _generate_single_level_base_anchors_hbb(self, sizes, device):
        anchors = []
        for size in sizes:
            area = float(size * size)
            for ratio in self.aspect_ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                anchors.append([
                    -0.5 * w, -0.5 * h,
                     0.5 * w,  0.5 * h
                ])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def _generate_single_level_base_anchors_obb(self, sizes, device):
        anchors = []
        for size in sizes:
            area = float(size * size)
            for ratio in self.aspect_ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                for angle in self.angles:
                    anchors.append([0.0, 0.0, w, h, angle])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def _grid_anchors_hbb(self, feat_size, stride, base_anchors, device):
        feat_h, feat_w = feat_size

        shifts_x = (torch.arange(feat_w, device=device) + self.offset) * stride
        shifts_y = (torch.arange(feat_h, device=device) + self.offset) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shifts = torch.stack(
            [
                shift_x.reshape(-1),
                shift_y.reshape(-1),
                shift_x.reshape(-1),
                shift_y.reshape(-1),
            ],
            dim=1,
        )

        anchors = (base_anchors[None, :, :] + shifts[:, None, :]).reshape(-1, 4)
        return anchors

    def _grid_anchors_obb(self, feat_size, stride, base_anchors, device):
        feat_h, feat_w = feat_size

        shifts_x = (torch.arange(feat_w, device=device) + self.offset) * stride
        shifts_y = (torch.arange(feat_h, device=device) + self.offset) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shifts = torch.stack(
            [
                shift_x.reshape(-1),
                shift_y.reshape(-1),
            ],
            dim=1,
        )

        base = base_anchors[None, :, :].repeat(shifts.shape[0], 1, 1)
        base[:, :, 0] += shifts[:, None, 0]
        base[:, :, 1] += shifts[:, None, 1]
        return base.reshape(-1, 5)

    def grid_priors(self, featmap_sizes, device):
        """
        Generate anchors for each level.

        Args:
            featmap_sizes (list[tuple[int, int]])
            device

        Returns:
            list[Tensor]
        """
        mlvl_anchors = []

        for sizes, stride, feat_size in zip(self.sizes, self.strides, featmap_sizes):
            if self.mode == "hbb":
                base_anchors = self._generate_single_level_base_anchors_hbb(sizes, device)
                anchors = self._grid_anchors_hbb(feat_size, stride, base_anchors, device)
            else:
                base_anchors = self._generate_single_level_base_anchors_obb(sizes, device)
                anchors = self._grid_anchors_obb(feat_size, stride, base_anchors, device)

            mlvl_anchors.append(anchors)

        return mlvl_anchors

    def valid_flags(self, featmap_sizes, padded_image_size, device):
        """
        Compute valid flags for each level.

        Args:
            featmap_sizes (list[tuple[int, int]])
            padded_image_size (tuple[int, int]):
                (height, width)
        """
        img_h, img_w = padded_image_size
        multi_level_flags = []

        for feat_size, stride in zip(featmap_sizes, self.strides):
            feat_h, feat_w = feat_size
            valid_h = min(int(math.ceil(img_h / stride)), feat_h)
            valid_w = min(int(math.ceil(img_w / stride)), feat_w)

            flags = torch.zeros((feat_h, feat_w), dtype=torch.bool, device=device)
            flags[:valid_h, :valid_w] = True

            num_base = self.num_anchors_per_location()[len(multi_level_flags)]
            flags = flags.reshape(-1, 1).expand(flags.numel(), num_base).reshape(-1)
            multi_level_flags.append(flags)

        return multi_level_flags

    def get_anchors(self, featmap_sizes, img_metas, device):
        """
        Generate anchors and valid flags for each image.

        Args:
            featmap_sizes (list[tuple[int, int]])
            img_metas (list[dict])
            device

        Returns:
            tuple:
                anchor_list, valid_flag_list
        """
        multi_level_anchors = self.grid_priors(featmap_sizes, device=device)

        anchor_list = [multi_level_anchors for _ in range(len(img_metas))]
        valid_flag_list = []

        for img_meta in img_metas:
            if "pad_shape" in img_meta:
                pad_h, pad_w = img_meta["pad_shape"][:2]
            else:
                pad_h, pad_w = img_meta["img_shape"][:2]

            valid_flags = self.valid_flags(featmap_sizes, (pad_h, pad_w), device)
            valid_flag_list.append(valid_flags)

        return anchor_list, valid_flag_list
