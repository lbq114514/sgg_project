import torch
from sgg.structures.boxes import BoxList


def box_iou_hbb(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def pair_geometry_features(boxlist: BoxList) -> torch.Tensor:
    boxes = boxlist.boxes
    n = boxes.shape[0]
    centers = boxlist.centers()
    areas = boxlist.areas().clamp(min=1e-6)

    ci = centers[:, None, :].expand(n, n, 2)
    cj = centers[None, :, :].expand(n, n, 2)
    delta = cj - ci
    dist = torch.norm(delta, dim=-1, keepdim=True)

    ai = areas[:, None].expand(n, n).unsqueeze(-1)
    aj = areas[None, :].expand(n, n).unsqueeze(-1)
    area_ratio = torch.log(aj / ai)

    if boxlist.mode in {"xyxy", "xywh"}:
        bi = boxes[:, None, :].expand(n, n, 4)
        bj = boxes[None, :, :].expand(n, n, 4)
        wh_i = (bi[..., 2:4] - bi[..., 0:2]).clamp(min=1e-6)
        wh_j = (bj[..., 2:4] - bj[..., 0:2]).clamp(min=1e-6)
        scale = torch.log(wh_j / wh_i)
        return torch.cat([delta, dist, area_ratio, scale], dim=-1)

    ai_ang = boxes[:, 4][:, None].expand(n, n).unsqueeze(-1)
    aj_ang = boxes[:, 4][None, :].expand(n, n).unsqueeze(-1)
    angle_delta = (aj_ang - ai_ang) / 180.0
    wh_i = boxes[:, 2:4][:, None, :].expand(n, n, 2).clamp(min=1e-6)
    wh_j = boxes[None, :, 2:4].expand(n, n, 2).clamp(min=1e-6)
    scale = torch.log(wh_j / wh_i)
    return torch.cat([delta, dist, area_ratio, scale, angle_delta], dim=-1)
