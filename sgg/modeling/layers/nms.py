from typing import Optional, Tuple

import torch

try:
    from mmcv.ops import nms_rotated as mmcv_nms_rotated
except Exception:
    mmcv_nms_rotated = None

try:
    from mmrotate.core import obb2xyxy
except Exception:
    obb2xyxy = None


def _box_iou_hbb(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def _nms_hbb(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(int(current.item()))
        if order.numel() == 1:
            break
        ious = _box_iou_hbb(boxes[current : current + 1], boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _batched_nms_hbb(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    keep = []
    for label in torch.unique(idxs):
        inds = torch.nonzero(idxs == label, as_tuple=False).squeeze(1)
        if inds.numel() == 0:
            continue
        keep.append(inds[_nms_hbb(boxes[inds], scores[inds], iou_threshold)])
    if not keep:
        return boxes.new_zeros((0,), dtype=torch.long)
    keep = torch.cat(keep, dim=0)
    return keep[scores[keep].argsort(descending=True)]


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    mode: str = "hbb",
    obb_fallback_to_hbb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified NMS wrapper for HBB / OBB.

    Args:
        boxes (Tensor):
            HBB mode: shape (N, 4) in xyxy format
            OBB mode: shape (N, 5) in xywha format

        scores (Tensor):
            Shape (N,)

        iou_threshold (float):
            IoU threshold for NMS.

        mode (str):
            "hbb" or "obb"

        obb_fallback_to_hbb (bool):
            If True and ``mode="obb"`` but ``mmcv.ops.nms_rotated`` is not
            available, fall back to horizontal NMS by converting rotated boxes
            to their axis-aligned envelopes.

    Returns:
        tuple:
            dets (Tensor):
                HBB mode: (K, 5) -> [x1, y1, x2, y2, score]
                OBB mode: (K, 6) -> [cx, cy, w, h, a, score]

            keep (Tensor):
                Indices of kept boxes in the original input.
    """
    if boxes.numel() == 0:
        if mode == "hbb":
            return boxes.new_zeros((0, 5)), boxes.new_zeros((0,), dtype=torch.long)
        elif mode == "obb":
            return boxes.new_zeros((0, 6)), boxes.new_zeros((0,), dtype=torch.long)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if mode == "hbb":
        keep = _nms_hbb(boxes, scores, iou_threshold)
        dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
        return dets, keep

    if mode == "obb":
        if mmcv_nms_rotated is None:
            if not obb_fallback_to_hbb:
                raise ImportError(
                    "mmcv.ops.nms_rotated is required for OBB NMS. "
                    "Install mmcv-full or call with obb_fallback_to_hbb=True "
                    "to use axis-aligned fallback NMS."
                )
            hbb = _obb_to_hbb(boxes)
            keep = _nms_hbb(hbb, scores, iou_threshold)
            dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
            return dets, keep
        dets, keep = mmcv_nms_rotated(boxes, scores, iou_threshold)
        return dets, keep

    raise ValueError(f"Unsupported mode: {mode}")


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
    mode: str = "hbb",
    obb_fallback_to_hbb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched NMS wrapper for HBB / OBB.

    Args:
        boxes (Tensor):
            HBB mode: (N, 4)
            OBB mode: (N, 5)

        scores (Tensor):
            (N,)

        idxs (Tensor):
            Category ids or level ids, shape (N,)

        iou_threshold (float):
            IoU threshold.

        mode (str):
            "hbb" or "obb"

        obb_fallback_to_hbb (bool):
            If True and ``mode="obb"`` but ``mmcv.ops.nms_rotated`` is not
            available, fall back to horizontal batched NMS using the
            axis-aligned envelopes of oriented boxes.

    Returns:
        tuple:
            dets (Tensor)
            keep (Tensor)
    """
    if boxes.numel() == 0:
        if mode == "hbb":
            return boxes.new_zeros((0, 5)), boxes.new_zeros((0,), dtype=torch.long)
        elif mode == "obb":
            return boxes.new_zeros((0, 6)), boxes.new_zeros((0,), dtype=torch.long)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if mode == "hbb":
        keep = _batched_nms_hbb(boxes, scores, idxs, iou_threshold)
        dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
        return dets, keep

    if mode == "obb":
        if mmcv_nms_rotated is None:
            if not obb_fallback_to_hbb:
                raise ImportError(
                    "mmcv.ops.nms_rotated is required for OBB batched NMS. "
                    "Install mmcv-full or call with obb_fallback_to_hbb=True "
                    "to use axis-aligned fallback batched NMS."
                )
            hbb = _obb_to_hbb(boxes)
            keep = _batched_nms_hbb(hbb, scores, idxs, iou_threshold)
            dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
            return dets, keep

        unique_labels = torch.unique(idxs)
        all_keep = []

        for label in unique_labels:
            inds = torch.nonzero(idxs == label, as_tuple=False).squeeze(1)
            if inds.numel() == 0:
                continue

            dets_i, keep_i = mmcv_nms_rotated(boxes[inds], scores[inds], iou_threshold)
            original_keep = inds[keep_i]
            all_keep.append(original_keep)

        if len(all_keep) == 0:
            return boxes.new_zeros((0, 6)), boxes.new_zeros((0,), dtype=torch.long)

        keep = torch.cat(all_keep, dim=0)
        _, order = scores[keep].sort(descending=True)
        keep = keep[order]

        dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
        return dets, keep

    raise ValueError(f"Unsupported mode: {mode}")


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float,
    iou_threshold: float,
    max_num: int = -1,
    mode: str = "hbb",
    scores_include_bg: bool = False,
    bg_index: int = -1,
    obb_fallback_to_hbb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-class NMS for HBB / OBB.

    Args:
        boxes (Tensor):
            HBB mode:
                - shape (N, 4), shared boxes for all classes, or
                - shape (N, C*4), class-specific boxes

            OBB mode:
                - shape (N, 5), shared boxes for all classes, or
                - shape (N, C*5), class-specific boxes

        scores (Tensor):
            Shape (N, C), usually excluding background.

        score_threshold (float):
            Score threshold before NMS.

        iou_threshold (float):
            IoU threshold for NMS.

        max_num (int):
            Maximum number of kept detections.

        mode (str):
            "hbb" or "obb"

        scores_include_bg (bool):
            Whether ``scores`` includes a background column.

        bg_index (int):
            Index of the background column when ``scores_include_bg=True``.
            Defaults to ``-1`` for the common layout where background is last.

        obb_fallback_to_hbb (bool):
            Passed through to ``batched_nms`` for OBB fallback behavior.

    Returns:
        tuple:
            dets (Tensor)
            labels (Tensor)
    """
    if scores_include_bg:
        if scores.shape[1] == 0:
            det_dim = 5 if mode == "hbb" else 6
            return boxes.new_zeros((0, det_dim)), boxes.new_zeros((0,), dtype=torch.long)
        bg_index = bg_index if bg_index >= 0 else scores.shape[1] + bg_index
        if bg_index < 0 or bg_index >= scores.shape[1]:
            raise ValueError(f"bg_index {bg_index} is out of range for scores with shape {scores.shape}")
        keep_cols = [i for i in range(scores.shape[1]) if i != bg_index]
        scores = scores[:, keep_cols]

    num_classes = scores.shape[1]
    all_boxes = []
    all_scores = []
    all_labels = []

    box_dim = 4 if mode == "hbb" else 5

    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        valid_mask = cls_scores > score_threshold
        if valid_mask.sum() == 0:
            continue

        cls_scores = cls_scores[valid_mask]

        if boxes.shape[1] == box_dim:
            cls_boxes = boxes[valid_mask]
        else:
            cls_boxes = boxes[valid_mask, cls_id * box_dim:(cls_id + 1) * box_dim]

        cls_labels = boxes.new_full(
            (cls_boxes.shape[0],),
            cls_id,
            dtype=torch.long
        )

        all_boxes.append(cls_boxes)
        all_scores.append(cls_scores)
        all_labels.append(cls_labels)

    if len(all_boxes) == 0:
        det_dim = 5 if mode == "hbb" else 6
        return boxes.new_zeros((0, det_dim)), boxes.new_zeros((0,), dtype=torch.long)

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    dets, keep = batched_nms(
        boxes=boxes,
        scores=scores,
        idxs=labels,
        iou_threshold=iou_threshold,
        mode=mode,
        obb_fallback_to_hbb=obb_fallback_to_hbb,
    )

    labels = labels[keep]

    if max_num > 0:
        dets = dets[:max_num]
        labels = labels[:max_num]

    return dets, labels


def _obb_to_hbb(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``xywha`` boxes to axis-aligned ``xyxy`` envelopes."""
    if obb2xyxy is not None:
        return obb2xyxy(boxes, version="oc")

    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    angles = boxes[:, 4]

    cos = torch.abs(torch.cos(angles))
    sin = torch.abs(torch.sin(angles))
    half_w = 0.5 * (widths * cos + heights * sin)
    half_h = 0.5 * (widths * sin + heights * cos)

    x1 = ctr_x - half_w
    y1 = ctr_y - half_h
    x2 = ctr_x + half_w
    y2 = ctr_y + half_h
    return torch.stack([x1, y1, x2, y2], dim=1)
