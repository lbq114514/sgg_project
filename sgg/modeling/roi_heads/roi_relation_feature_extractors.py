from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.modeling.core.obb_ops import (
    angle_from_radians,
    angle_to_radians,
    get_boxlist_angle_unit,
    norm_angle,
    obb2poly,
    set_boxlist_angle_unit,
)
from sgg.modeling.roi_extractors import MultiScaleROIExtractor, RotatedROIExtractor
from sgg.structures.boxes import BoxList

def _average_obb_angle(angle1: torch.Tensor, angle2: torch.Tensor, angle_unit: str) -> torch.Tensor:
    rad1 = angle_to_radians(angle1, angle_unit)
    rad2 = angle_to_radians(angle2, angle_unit)
    sin2 = torch.sin(2.0 * rad1) + torch.sin(2.0 * rad2)
    cos2 = torch.cos(2.0 * rad1) + torch.cos(2.0 * rad2)
    avg = 0.5 * torch.atan2(sin2, cos2)
    return angle_from_radians(avg, angle_unit)


def _enclosing_union_obb(
    subject_boxes: torch.Tensor,
    object_boxes: torch.Tensor,
    angle_version: str,
    angle_unit: str,
) -> torch.Tensor:
    if subject_boxes.numel() == 0:
        return subject_boxes.new_zeros((0, 5))

    subj_poly = obb2poly(subject_boxes, version=angle_version, angle_unit=angle_unit).view(-1, 4, 2)
    obj_poly = obb2poly(object_boxes, version=angle_version, angle_unit=angle_unit).view(-1, 4, 2)
    points = torch.cat([subj_poly, obj_poly], dim=1)

    angle_candidates = torch.stack(
        [
            subject_boxes[:, 4],
            object_boxes[:, 4],
            _average_obb_angle(subject_boxes[:, 4], object_boxes[:, 4], angle_unit),
        ],
        dim=1,
    )
    right_angle = torch.as_tensor(np.pi / 2 if angle_unit == "radian" else 90.0, device=subject_boxes.device, dtype=subject_boxes.dtype)

    best_area = None
    best_boxes = None
    for cand_idx in range(angle_candidates.size(1)):
        angle = angle_candidates[:, cand_idx]
        rad = angle_to_radians(angle, angle_unit)
        cos = torch.cos(rad)
        sin = torch.sin(rad)
        rot = torch.stack(
            [
                torch.stack([cos, sin], dim=1),
                torch.stack([-sin, cos], dim=1),
            ],
            dim=1,
        )
        local = torch.bmm(points, rot)
        xy_min = local.min(dim=1).values
        xy_max = local.max(dim=1).values
        wh = (xy_max - xy_min).clamp(min=1e-4)
        ctr_local = 0.5 * (xy_min + xy_max)

        inv_rot = torch.stack(
            [
                torch.stack([cos, -sin], dim=1),
                torch.stack([sin, cos], dim=1),
            ],
            dim=1,
        )
        ctr_global = torch.bmm(ctr_local.unsqueeze(1), inv_rot).squeeze(1)
        boxes = torch.cat([ctr_global, wh, angle.unsqueeze(1)], dim=1)

        swap = boxes[:, 2] < boxes[:, 3]
        if swap.any():
            swapped = boxes[swap, 2].clone()
            boxes[swap, 2] = boxes[swap, 3]
            boxes[swap, 3] = swapped
            boxes[swap, 4] = boxes[swap, 4] + right_angle
        boxes[:, 4] = norm_angle(boxes[:, 4], angle_version, angle_unit=angle_unit)

        area = boxes[:, 2] * boxes[:, 3]
        if best_area is None:
            best_area = area
            best_boxes = boxes
        else:
            keep = area < best_area
            best_area = torch.where(keep, area, best_area)
            best_boxes = torch.where(keep[:, None], boxes, best_boxes)

    return best_boxes


def _legacy_obb2poly_le90_rad(rboxes: torch.Tensor) -> torch.Tensor:
    """Original RPCM ``obb2poly_le90`` implementation.

    The legacy code assumes angle values are already radians and applies
    ``sin/cos`` directly.  Keep this helper local to the legacy path so newer
    callers continue using explicit angle-unit conversion.
    """
    if rboxes.numel() == 0:
        return rboxes.new_zeros((0, 8))
    x_ctr, y_ctr, width, height, angle = rboxes.t()
    tl_x, tl_y = -width * 0.5, -height * 0.5
    br_x, br_y = width * 0.5, height * 0.5
    rects = torch.stack(
        [tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
        dim=0,
    ).reshape(2, 4, rboxes.size(0)).permute(2, 0, 1)
    sin_a, cos_a = torch.sin(angle), torch.cos(angle)
    rot = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=0).reshape(
        2, 2, rboxes.size(0)
    ).permute(2, 0, 1)
    polys = rot.matmul(rects).permute(2, 1, 0).reshape(-1, rboxes.size(0)).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def _legacy_poly2obb_le90_8_batch(polys: torch.Tensor) -> torch.Tensor:
    """Original RPCM ``poly2obb_np_le90_8_batch_vectorized`` behavior."""
    if polys.numel() == 0:
        return polys.new_zeros((0, 5))
    polys_np = polys.detach().cpu().reshape((-1, 8, 2)).to(torch.float32).numpy()
    rbboxes = np.array([cv2.minAreaRect(poly) for poly in polys_np], dtype=object)
    x = np.array([item[0][0] for item in rbboxes])
    y = np.array([item[0][1] for item in rbboxes])
    w = np.array([item[1][0] for item in rbboxes])
    h = np.array([item[1][1] for item in rbboxes])
    angle = np.array([item[2] for item in rbboxes]) / 180.0 * np.pi

    swap = w < h
    w[swap], h[swap] = h[swap], w[swap]
    angle[swap] += np.pi / 2.0

    too_large = angle >= np.pi / 2.0
    angle[too_large] -= np.pi
    too_small = angle < -np.pi / 2.0
    angle[too_small] += np.pi

    boxes = np.column_stack((x, y, w, h, angle))
    return torch.as_tensor(boxes, dtype=polys.dtype, device=polys.device)


def _legacy_union_obb(
    subject_boxes: torch.Tensor,
    object_boxes: torch.Tensor,
    angle_unit: str,
) -> torch.Tensor:
    """Match original RPCM ``boxlist_union(..., flag2=True)`` for OBBs."""
    if subject_boxes.numel() == 0:
        return subject_boxes.new_zeros((0, 5))
    subj_rad = subject_boxes.clone()
    obj_rad = object_boxes.clone()
    subj_rad[:, 4] = angle_to_radians(subj_rad[:, 4], angle_unit)
    obj_rad[:, 4] = angle_to_radians(obj_rad[:, 4], angle_unit)
    subj_poly = _legacy_obb2poly_le90_rad(subj_rad)
    obj_poly = _legacy_obb2poly_le90_rad(obj_rad)
    union_rad = _legacy_poly2obb_le90_8_batch(torch.cat((subj_poly, obj_poly), dim=1))
    union_boxes = union_rad.to(dtype=subject_boxes.dtype, device=subject_boxes.device)
    union_boxes[:, 4] = angle_from_radians(union_boxes[:, 4], angle_unit)
    return union_boxes


class RelationBoxFeatureExtractor(nn.Module):
    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        model_cfg = cfg["MODEL"]
        roi_cfg = dict(model_cfg.get("ROI_EXTRACTOR", {}))
        pool_size = int(roi_cfg.get("POOL_SIZE", 7))
        box_mode = model_cfg.get("BOX_MODE", "obb")
        roi_cfg["OUT_CHANNELS"] = in_channels
        if roi_cfg.get("FEATMAP_NAMES") or roi_cfg.get("FEATMAP_STRIDES"):
            roi_cfg.setdefault("MODE", "obb" if box_mode == "obb" else "hbb")
            self.pooler = MultiScaleROIExtractor.from_config(roi_cfg)
        else:
            if box_mode == "obb":
                self.pooler = RotatedROIExtractor.from_config(roi_cfg)
            else:
                self.pooler = MultiScaleROIExtractor.from_config(
                    {
                        "POOL_SIZE": pool_size,
                        "OUT_CHANNELS": in_channels,
                        "MODE": "hbb",
                        "FEATMAP_NAMES": ["p2", "p3", "p4", "p5"],
                        "FEATMAP_STRIDES": [4, 8, 16, 32],
                    }
                )
        hidden_dim = int(cfg["MODEL"]["ROI_RELATION_HEAD"].get("CONTEXT_POOLING_DIM", cfg["MODEL"]["RELATION_HEAD"]["NODE_DIM"]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pool_size * pool_size, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = hidden_dim

    def forward(self, features, proposals: Sequence[BoxList]):
        pooled = self.pooler(features, list(proposals))
        if pooled.numel() == 0:
            return pooled.new_zeros((0, self.out_channels))
        return self.head(pooled)


class RelationUnionFeatureExtractor(nn.Module):
    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        roi_cfg = dict(cfg["MODEL"].get("ROI_EXTRACTOR", {}))
        rel_cfg = cfg["MODEL"].get("ROI_RELATION_HEAD", {})
        box_mode = cfg["MODEL"].get("BOX_MODE", "hbb")
        pool_size = int(roi_cfg.get("POOL_SIZE", 7))
        roi_cfg["OUT_CHANNELS"] = in_channels
        self.angle_version = roi_cfg.get("ANGLE_VERSION", "oc")
        self.roi_align_chunk_size = int(rel_cfg.get("UNION_ROI_ALIGN_CHUNK_SIZE", 2048))
        featmap_names = roi_cfg.get("FEATMAP_NAMES", ["p2", "p3", "p4", "p5"])
        featmap_strides = roi_cfg.get("FEATMAP_STRIDES", [4, 8, 16, 32])
        self.featmap_names = featmap_names
        self.featmap_strides = featmap_strides
        if roi_cfg.get("FEATMAP_NAMES") or roi_cfg.get("FEATMAP_STRIDES"):
            roi_cfg["MODE"] = "obb" if box_mode == "obb" else "hbb"
            roi_cfg.setdefault("FEATMAP_NAMES", featmap_names)
            roi_cfg.setdefault("FEATMAP_STRIDES", featmap_strides)
            self.pooler = MultiScaleROIExtractor.from_config(roi_cfg)
        else:
            if box_mode == "obb":
                self.pooler = RotatedROIExtractor.from_config(roi_cfg)
            else:
                roi_cfg["MODE"] = "hbb"
                roi_cfg.setdefault("FEATMAP_NAMES", featmap_names)
                roi_cfg.setdefault("FEATMAP_STRIDES", featmap_strides)
                self.pooler = MultiScaleROIExtractor.from_config(roi_cfg)
        self.box_mode = box_mode
        self.separate_spatial = bool(rel_cfg.get("CAUSAL", {}).get("SEPARATE_SPATIAL", False))
        self.use_rpcm_legacy_union = bool(rel_cfg.get("RPCM_LEGACY_UNION_BOX", False))
        hidden_dim = int(cfg["MODEL"]["ROI_RELATION_HEAD"].get("CONTEXT_POOLING_DIM", cfg["MODEL"]["RELATION_HEAD"]["EDGE_DIM"]))
        detector_feat_dim = int(cfg["MODEL"].get("ROI_BOX_HEAD", {}).get("MLP_HEAD_DIM", hidden_dim))
        self.pool_size = pool_size
        self.rect_size = pool_size * 4 - 1
        self.rect_conv = nn.Sequential(
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pool_size * pool_size, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.detector_head_proj = (
            nn.Linear(detector_feat_dim, hidden_dim)
            if detector_feat_dim != hidden_dim
            else nn.Identity()
        )
        if self.separate_spatial:
            self.spatial_fc = nn.Sequential(
                nn.Linear(in_channels * pool_size * pool_size, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(inplace=True),
            )
        self.out_channels = hidden_dim

    def _resize_proposal_for_rect(self, proposal: BoxList) -> BoxList:
        if len(proposal) == 0:
            box_dim = proposal.bbox.size(1)
            resized = BoxList(proposal.bbox.new_zeros((0, box_dim)), (self.rect_size, self.rect_size), proposal.mode)
            if proposal.mode == "xywha":
                set_boxlist_angle_unit(resized, get_boxlist_angle_unit(proposal))
            return resized
        width, height = proposal.size
        ratio_width = float(self.rect_size) / float(width)
        ratio_height = float(self.rect_size) / float(height)
        if proposal.mode == "xywha":
            scaled_box = proposal.bbox.clone()
            scaled_box[:, 0] *= ratio_width
            scaled_box[:, 1] *= ratio_height
            scaled_box[:, 2] *= np.sqrt(ratio_width * ratio_height)
            scaled_box[:, 3] *= np.sqrt(ratio_width * ratio_height)
            resized = BoxList(scaled_box, (self.rect_size, self.rect_size), "xywha")
            set_boxlist_angle_unit(resized, get_boxlist_angle_unit(proposal))
            return resized
        scaled = proposal.convert("xyxy").bbox.clone()
        scaled[:, 0] *= ratio_width
        scaled[:, 2] *= ratio_width
        scaled[:, 1] *= ratio_height
        scaled[:, 3] *= ratio_height
        return BoxList(scaled, (self.rect_size, self.rect_size), "xyxy")

    def _encode_obb_rectangles(
        self,
        head_boxes: torch.Tensor,
        tail_boxes: torch.Tensor,
        device: torch.device,
        angle_unit: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rel = head_boxes.size(0)
        if num_rel == 0:
            empty = torch.zeros((0, self.rect_size, self.rect_size), device=device)
            return empty, empty
        if self.use_rpcm_legacy_union:
            head_boxes_rad = head_boxes.clone()
            tail_boxes_rad = tail_boxes.clone()
            head_boxes_rad[:, 4] = angle_to_radians(head_boxes_rad[:, 4], angle_unit)
            tail_boxes_rad[:, 4] = angle_to_radians(tail_boxes_rad[:, 4], angle_unit)
            head_polys = _legacy_obb2poly_le90_rad(head_boxes_rad).view(-1, 4, 2).detach().cpu().numpy()
            tail_polys = _legacy_obb2poly_le90_rad(tail_boxes_rad).view(-1, 4, 2).detach().cpu().numpy()
        else:
            head_polys = obb2poly(head_boxes, version=self.angle_version, angle_unit=angle_unit).view(-1, 4, 2).detach().cpu().numpy()
            tail_polys = obb2poly(tail_boxes, version=self.angle_version, angle_unit=angle_unit).view(-1, 4, 2).detach().cpu().numpy()
        head_centers = head_boxes[:, :2].detach().cpu().numpy()
        tail_centers = tail_boxes[:, :2].detach().cpu().numpy()
        head_x = np.where(head_polys[:, :, 0] <= head_centers[:, None, 0], np.floor(head_polys[:, :, 0]), np.ceil(head_polys[:, :, 0]))
        head_y = np.where(head_polys[:, :, 1] <= head_centers[:, None, 1], np.floor(head_polys[:, :, 1]), np.ceil(head_polys[:, :, 1]))
        tail_x = np.where(tail_polys[:, :, 0] <= tail_centers[:, None, 0], np.floor(tail_polys[:, :, 0]), np.ceil(tail_polys[:, :, 0]))
        tail_y = np.where(tail_polys[:, :, 1] <= tail_centers[:, None, 1], np.floor(tail_polys[:, :, 1]), np.ceil(tail_polys[:, :, 1]))
        head_pts = np.stack((head_x, head_y), axis=-1).astype(np.int32)
        tail_pts = np.stack((tail_x, tail_y), axis=-1).astype(np.int32)
        head_rect = torch.zeros((num_rel, self.rect_size, self.rect_size), dtype=torch.float32)
        tail_rect = torch.zeros((num_rel, self.rect_size, self.rect_size), dtype=torch.float32)
        for idx in range(num_rel):
            head_rect[idx] = torch.from_numpy(cv2.fillPoly(np.zeros((self.rect_size, self.rect_size), dtype=np.float32), [head_pts[idx]], 1.0))
            tail_rect[idx] = torch.from_numpy(cv2.fillPoly(np.zeros((self.rect_size, self.rect_size), dtype=np.float32), [tail_pts[idx]], 1.0))
        return head_rect.to(device), tail_rect.to(device)

    def _encode_hbb_rectangles(self, head_boxes: torch.Tensor, tail_boxes: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        num_rel = head_boxes.size(0)
        dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
        dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
        head_rect = (
            (dummy_x_range >= head_boxes[:, 0].floor().view(-1, 1, 1).long())
            & (dummy_x_range <= head_boxes[:, 2].ceil().view(-1, 1, 1).long())
            & (dummy_y_range >= head_boxes[:, 1].floor().view(-1, 1, 1).long())
            & (dummy_y_range <= head_boxes[:, 3].ceil().view(-1, 1, 1).long())
        ).float()
        tail_rect = (
            (dummy_x_range >= tail_boxes[:, 0].floor().view(-1, 1, 1).long())
            & (dummy_x_range <= tail_boxes[:, 2].ceil().view(-1, 1, 1).long())
            & (dummy_y_range >= tail_boxes[:, 1].floor().view(-1, 1, 1).long())
            & (dummy_y_range <= tail_boxes[:, 3].ceil().view(-1, 1, 1).long())
        ).float()
        return head_rect, tail_rect

    def forward(self, features, proposals: Sequence[BoxList], rel_pair_idxs, OBj=None):
        feature_ref = features[next(iter(features))] if isinstance(features, dict) else (
            features[0] if isinstance(features, (list, tuple)) else features
        )
        feature_device = feature_ref.device
        union_proposals = []
        rect_inputs = []
        for proposal, pair_idx in zip(proposals, rel_pair_idxs):
            if pair_idx.numel() == 0:
                box_dim = 5 if self.box_mode == "obb" else 4
                mode = "xywha" if self.box_mode == "obb" else "xyxy"
                union_proposals.append(BoxList(proposal.bbox.new_zeros((0, box_dim)), proposal.size, mode))
                if mode == "xywha":
                    set_boxlist_angle_unit(union_proposals[-1], get_boxlist_angle_unit(proposal))
                continue
            head_proposal = proposal[pair_idx[:, 0]]
            tail_proposal = proposal[pair_idx[:, 1]]
            if proposal.mode == "xywha":
                angle_unit = get_boxlist_angle_unit(proposal)
                if self.use_rpcm_legacy_union:
                    union_boxes = _legacy_union_obb(head_proposal.bbox, tail_proposal.bbox, angle_unit)
                else:
                    union_boxes = _enclosing_union_obb(head_proposal.bbox, tail_proposal.bbox, self.angle_version, angle_unit)
                union_boxlist = BoxList(union_boxes, proposal.size, "xywha")
                set_boxlist_angle_unit(union_boxlist, angle_unit)
                union_proposals.append(union_boxlist)
            else:
                union_boxes = torch.stack(
                    [
                        torch.min(head_proposal.bbox[:, 0], tail_proposal.bbox[:, 0]),
                        torch.min(head_proposal.bbox[:, 1], tail_proposal.bbox[:, 1]),
                        torch.max(head_proposal.bbox[:, 2], tail_proposal.bbox[:, 2]),
                        torch.max(head_proposal.bbox[:, 3], tail_proposal.bbox[:, 3]),
                    ],
                    dim=1,
                )
                union_proposals.append(BoxList(union_boxes, proposal.size, "xyxy"))
            head_rect_prop = self._resize_proposal_for_rect(head_proposal)
            tail_rect_prop = self._resize_proposal_for_rect(tail_proposal)
            if proposal.mode == "xywha":
                head_rect, tail_rect = self._encode_obb_rectangles(
                    head_rect_prop.bbox,
                    tail_rect_prop.bbox,
                    feature_device,
                    get_boxlist_angle_unit(proposal),
                )
            else:
                head_rect, tail_rect = self._encode_hbb_rectangles(head_rect_prop.bbox, tail_rect_prop.bbox, feature_device)
            rect_inputs.append(torch.stack((head_rect, tail_rect), dim=1))

        total_unions = sum(len(p) for p in union_proposals)
        if total_unions == 0:
            return feature_ref.new_zeros((0, self.out_channels))
        rect_inputs = torch.cat(rect_inputs, dim=0) if rect_inputs else torch.zeros((0, 2, self.rect_size, self.rect_size), device=feature_device)
        rect_features = self.rect_conv(rect_inputs)

        if OBj is not None and hasattr(OBj, "bbox_roi_extractor") and hasattr(OBj, "bbox_head"):
            union_vis_features = OBj.bbox_roi_extractor(features, union_proposals)
            if self.separate_spatial:
                region_features = self.detector_head_proj(OBj.bbox_head(union_vis_features))
                spatial_features = self.spatial_fc(rect_features.flatten(1))
                return (region_features, spatial_features)
            union_features = self.detector_head_proj(OBj.bbox_head(union_vis_features + rect_features))
        else:
            union_vis_features = self.pooler(features, union_proposals)
            if self.separate_spatial:
                region_features = self.head(union_vis_features)
                spatial_features = self.spatial_fc(rect_features.flatten(1))
                return (region_features, spatial_features)
            union_features = self.head(union_vis_features + rect_features)
        return union_features


def make_roi_relation_feature_extractor(cfg: dict, in_channels: int):
    return RelationUnionFeatureExtractor(cfg, in_channels)


def make_roi_box_feature_extractor(cfg: dict, in_channels: int):
    return RelationBoxFeatureExtractor(cfg, in_channels)
