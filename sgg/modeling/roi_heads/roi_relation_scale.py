from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from sgg.modeling.core.obb_ops import angle_to_radians, get_boxlist_angle_unit


def proposal_box_tensor(proposal):
    if proposal.mode in {"xyxy", "xywh", "xywha"}:
        return proposal.bbox
    raise ValueError(f"Unsupported box mode: {proposal.mode}")


def box_geometry_from_tensor(boxes: torch.Tensor, mode: str):
    if boxes.numel() == 0:
        return (
            boxes.new_zeros((0, 2)),
            boxes.new_zeros((0, 2)),
            boxes.new_zeros((0, 1)),
        )
    if mode == "xywha":
        ctr = boxes[:, :2]
        wh = boxes[:, 2:4].clamp(min=1e-6)
        angle = boxes[:, 4:5]
    else:
        if mode == "xywh":
            x, y, w, h = boxes.unbind(dim=1)
            ctr = torch.stack([x + 0.5 * w, y + 0.5 * h], dim=1)
            wh = torch.stack([w, h], dim=1).clamp(min=1e-6)
        else:
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6)
            ctr = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        angle = boxes.new_zeros((boxes.size(0), 1))
    return ctr, wh, angle


def encode_pair_scale_info(proposals: Sequence, rel_pair_idxs) -> torch.Tensor:
    pair_infos = []
    for proposal, pair_idx in zip(proposals, rel_pair_idxs):
        if pair_idx.numel() == 0:
            continue
        boxes = proposal_box_tensor(proposal)
        ctr, wh, angle = box_geometry_from_tensor(boxes, proposal.mode)
        subj_ctr = ctr[pair_idx[:, 0]]
        obj_ctr = ctr[pair_idx[:, 1]]
        subj_wh = wh[pair_idx[:, 0]]
        obj_wh = wh[pair_idx[:, 1]]
        subj_area = (subj_wh[:, :1] * subj_wh[:, 1:2]).clamp(min=1e-6)
        obj_area = (obj_wh[:, :1] * obj_wh[:, 1:2]).clamp(min=1e-6)
        union_wh = torch.max(subj_ctr + 0.5 * subj_wh, obj_ctr + 0.5 * obj_wh) - torch.min(
            subj_ctr - 0.5 * subj_wh,
            obj_ctr - 0.5 * obj_wh,
        )
        union_area = (union_wh[:, :1] * union_wh[:, 1:2]).clamp(min=1e-6)
        center_dist = torch.norm(obj_ctr - subj_ctr, dim=1, keepdim=True)
        angle_delta = angle_to_radians(
            angle[pair_idx[:, 1]] - angle[pair_idx[:, 0]],
            get_boxlist_angle_unit(proposal),
        )
        scale_info = torch.cat(
            [
                torch.log(subj_area),
                torch.log(obj_area),
                torch.log(union_area),
                torch.log(obj_area / subj_area),
                center_dist / union_wh.norm(dim=1, keepdim=True).clamp(min=1e-6),
                torch.log(subj_wh[:, :1] / subj_wh[:, 1:2].clamp(min=1e-6)),
                torch.log(obj_wh[:, :1] / obj_wh[:, 1:2].clamp(min=1e-6)),
                torch.sin(angle_delta),
                torch.cos(angle_delta),
                union_wh[:, :1] / union_wh[:, 1:2].clamp(min=1e-6),
            ],
            dim=1,
        )
        pair_infos.append(scale_info)
    if not pair_infos:
        if proposals:
            return proposals[0].bbox.new_zeros((0, 10))
        return torch.zeros((0, 10))
    return torch.cat(pair_infos, dim=0)


class PairScaleEncoder(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, scale_info: torch.Tensor) -> torch.Tensor:
        if scale_info.numel() == 0:
            return scale_info.new_zeros((0, self.mlp[-1].out_features))
        return self.mlp(scale_info)
