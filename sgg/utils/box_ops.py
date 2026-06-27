from __future__ import annotations

import torch


def generalized_box_iou_hbb(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x11, y11, x12, y12 = boxes1.unbind(-1)
    x21, y21, x22, y22 = boxes2.unbind(-1)

    inter_x1 = torch.max(x11[:, None], x21[None])
    inter_y1 = torch.max(y11[:, None], y21[None])
    inter_x2 = torch.min(x12[:, None], x22[None])
    inter_y2 = torch.min(y12[:, None], y22[None])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1[:, None] + area2[None] - inter + 1e-6
    iou = inter / union

    enc_x1 = torch.min(x11[:, None], x21[None])
    enc_y1 = torch.min(y11[:, None], y21[None])
    enc_x2 = torch.max(x12[:, None], x22[None])
    enc_y2 = torch.max(y12[:, None], y22[None])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-6
    return iou - (enc_area - union) / enc_area
