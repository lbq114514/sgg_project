from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmcv.ops import RoIAlignRotated
except Exception:
    RoIAlignRotated = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgg.structures.boxlist import BoxList

def _convert_boxes_to_roi_format(
    boxes: Union[List[torch.Tensor], List["BoxList"]],
    mode: str = "hbb",
) -> torch.Tensor:
    """
    Convert per-image boxes into RoIAlign input format.

    HBB output format:
        (batch_idx, x1, y1, x2, y2)

    OBB output format:
        (batch_idx, cx, cy, w, h, angle)
    """
    concat_boxes = []
    device = None
    dtype = None

    for i, boxes_per_image in enumerate(boxes):
        if hasattr(boxes_per_image, "bbox"):
            boxes_tensor = boxes_per_image.bbox
        else:
            boxes_tensor = boxes_per_image

        if boxes_tensor.numel() == 0:
            continue

        device = boxes_tensor.device
        dtype = boxes_tensor.dtype

        img_inds = torch.full(
            (boxes_tensor.shape[0], 1),
            i,
            dtype=boxes_tensor.dtype,
            device=boxes_tensor.device,
        )
        rois = torch.cat([img_inds, boxes_tensor], dim=1)
        concat_boxes.append(rois)

    if len(concat_boxes) == 0:
        dim = 5 if mode == "hbb" else 6
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.zeros((0, dim), device=device, dtype=dtype)

    return torch.cat(concat_boxes, dim=0)


class ROIAlign(nn.Module):
    """
    Unified ROIAlign wrapper for HBB / OBB.

    HBB mode:
        Uses torchvision.ops.roi_align

    OBB mode:
        Uses mmcv.ops.RoIAlignRotated

    Args:
        output_size (int or tuple[int, int])
        spatial_scale (float)
        sampling_ratio (int)
        aligned (bool)
        mode (str):
            "hbb" or "obb"
        clockwise (bool):
            Used only in OBB mode. Must be consistent with your angle convention.
    """

    def __init__(
        self,
        output_size,
        spatial_scale,
        sampling_ratio=0,
        aligned=True,
        mode="hbb",
        clockwise=False,
        use_torchvision_for_hbb=False,
    ):
        super().__init__()

        if mode not in ("hbb", "obb"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.mode = mode
        self.clockwise = clockwise
        self.use_torchvision_for_hbb = use_torchvision_for_hbb

        if self.mode == "obb":
            if RoIAlignRotated is None:
                raise ImportError("mmcv.ops.RoIAlignRotated is required for OBB ROIAlign")

            self.obb_roi_align = RoIAlignRotated(
                out_size=output_size,
                spatial_scale=spatial_scale,
                sample_num=sampling_ratio,
                clockwise=clockwise,
            )
        else:
            self.obb_roi_align = None

    def forward(
        self,
        features: torch.Tensor,
        boxes: Union[torch.Tensor, List[torch.Tensor], List["BoxList"]],
    ) -> torch.Tensor:
        """
        Args:
            features (Tensor):
                Shape (N, C, H, W)

            boxes:
                HBB mode:
                    - Tensor of shape (K, 5), or
                    - list of per-image boxes in shape (Mi, 4), or
                    - list of BoxList with mode 'xyxy'

                OBB mode:
                    - Tensor of shape (K, 6), or
                    - list of per-image boxes in shape (Mi, 5), or
                    - list of BoxList with mode 'xywha'

        Returns:
            Tensor:
                ROI features of shape (K, C, output_h, output_w)
        """
        if isinstance(boxes, (list, tuple)):
            rois = _convert_boxes_to_roi_format(boxes, mode=self.mode)
        else:
            rois = boxes

        if self.mode == "hbb":
            if rois.numel() == 0:
                channels = features.shape[1]
                if isinstance(self.output_size, int):
                    out_h, out_w = self.output_size, self.output_size
                else:
                    out_h, out_w = self.output_size
                return features.new_zeros((0, channels, out_h, out_w))

            if self.use_torchvision_for_hbb:
                from torchvision.ops import roi_align as tv_roi_align
                return tv_roi_align(
                    input=features,
                    boxes=rois,
                    output_size=self.output_size,
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio,
                    aligned=self.aligned,
                )

            return self._roi_align_hbb_fallback(features, rois)

        if self.mode == "obb":
            if rois.numel() == 0:
                channels = features.shape[1]
                if isinstance(self.output_size, int):
                    out_h, out_w = self.output_size, self.output_size
                else:
                    out_h, out_w = self.output_size
                return features.new_zeros((0, channels, out_h, out_w))

            return self.obb_roi_align(features, rois)

        raise ValueError(f"Unsupported mode: {self.mode}")

    def _roi_align_hbb_fallback(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        if isinstance(self.output_size, int):
            out_h, out_w = self.output_size, self.output_size
        else:
            out_h, out_w = self.output_size

        outputs = []
        feat_h, feat_w = features.shape[-2:]
        for roi in rois:
            batch_idx = int(roi[0].item())
            x1, y1, x2, y2 = roi[1:] * self.spatial_scale

            if self.aligned:
                x1 = x1 - 0.5
                y1 = y1 - 0.5
                x2 = x2 - 0.5
                y2 = y2 - 0.5

            x1i = int(torch.floor(x1).item())
            y1i = int(torch.floor(y1).item())
            x2i = int(torch.ceil(x2).item())
            y2i = int(torch.ceil(y2).item())

            x1i = max(0, min(x1i, feat_w - 1))
            y1i = max(0, min(y1i, feat_h - 1))
            x2i = max(x1i + 1, min(x2i, feat_w))
            y2i = max(y1i + 1, min(y2i, feat_h))

            crop = features[batch_idx : batch_idx + 1, :, y1i:y2i, x1i:x2i]
            pooled = F.adaptive_avg_pool2d(crop, (out_h, out_w))
            outputs.append(pooled)

        return torch.cat(outputs, dim=0) if outputs else features.new_zeros((0, features.shape[1], out_h, out_w))


class MultiScaleROIAlign(nn.Module):
    """
    Simple multi-scale ROIAlign wrapper.

    This version applies ROIAlign level by level according to a user-provided
    level assignment tensor.

    Args:
        output_size
        sampling_ratio
        featmap_strides (Sequence[int])
        mode (str):
            "hbb" or "obb"
        aligned (bool)
        clockwise (bool):
            Used only for OBB mode
    """

    def __init__(
        self,
        output_size,
        sampling_ratio,
        featmap_strides: Sequence[int],
        mode="hbb",
        aligned=True,
        clockwise=False,
    ):
        super().__init__()

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.featmap_strides = featmap_strides
        self.mode = mode

        self.poolers = nn.ModuleList(
            [
                ROIAlign(
                    output_size=output_size,
                    spatial_scale=1.0 / s,
                    sampling_ratio=sampling_ratio,
                    aligned=aligned,
                    mode=mode,
                    clockwise=clockwise,
                )
                for s in featmap_strides
            ]
        )

    def forward(
        self,
        features: List[torch.Tensor],
        rois: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features (list[Tensor]):
                Multi-level feature maps.

            rois (Tensor):
                HBB mode: (K, 5)
                OBB mode: (K, 6)

            levels (Tensor):
                Long tensor of shape (K,), indicating which feature level
                each RoI should use.

        Returns:
            Tensor:
                Pooled ROI features.
        """
        if len(features) != len(self.poolers):
            raise ValueError("features and featmap_strides should have the same length")

        if rois.numel() == 0:
            channels = features[0].shape[1]
            if isinstance(self.output_size, int):
                out_h, out_w = self.output_size, self.output_size
            else:
                out_h, out_w = self.output_size
            return features[0].new_zeros((0, channels, out_h, out_w))

        num_rois = rois.shape[0]
        dtype = features[0].dtype
        device = features[0].device

        if isinstance(self.output_size, int):
            out_h, out_w = self.output_size, self.output_size
        else:
            out_h, out_w = self.output_size

        output = torch.zeros(
            (num_rois, features[0].shape[1], out_h, out_w),
            dtype=dtype,
            device=device,
        )

        for level, pooler in enumerate(self.poolers):
            inds = torch.nonzero(levels == level, as_tuple=False).squeeze(1)
            if inds.numel() == 0:
                continue
            pooled = pooler(features[level], rois[inds])
            output[inds] = pooled

        return output
