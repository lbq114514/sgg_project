from __future__ import annotations

import math
from typing import Dict, List, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.modeling.roi_heads.roi_relation_scale import (
    PairScaleEncoder,
    box_geometry_from_tensor as _box_geometry_from_tensor,
    encode_pair_scale_info as _encode_pair_scale_info,
    proposal_box_tensor as _proposal_box_tensor,
)
from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


def _cfg_get(cfg: dict, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

_DEFAULT_REL_ANTONYM_NAME_PAIRS = [
    ("parallelly docked at", "isolatedly docked at"),
    ("docking at the same dock with", "docking at the different dock with"),
    ("co-storage with", "not co-storage with"),
    ("run along", "not run along"),
    ("parking in the same apron with", "parking in the different apron with"),
    ("parked alongside with", "not parked alongside with"),
    ("approach", "away from"),
    ("within safe distance of", "within danger distance of"),
    ("in the same parking with", "in the different parking with"),
    ("docked alongside with", "not docked alongside with"),
    ("driving in the same direction with", "driving in the opposite direction with"),
    ("driving in the different lane with", "driving in the same lane with"),
    ("within different line of", "within same line of"),
    ("directly connected to", "indirectly connected to"),
    ("directly transmit electricity to", "indirectly transmit electricity to"),
    ("working on", "not working on"),
]

_DEFAULT_REL_COMPETITOR_NAME_PAIRS = [
    ("parking in the same apron with", "parked alongside with"),
    ("parking in the same apron with", "in the same parking with"),
    ("parking in the different apron with", "within safe distance of"),
    ("parking in the same apron with", "within safe distance of"),
    ("parked alongside with", "within safe distance of"),
    ("parked alongside with", "driving in the same direction with"),
    ("parked alongside with", "driving in the different lane with"),
    ("in the same parking with", "parked alongside with"),
    ("in the same parking with", "within safe distance of"),
    ("driving in the same direction with", "within safe distance of"),
    ("driving in the different lane with", "within safe distance of"),
    ("driving in the same lane with", "within safe distance of"),
    ("driving in the different lane with", "driving in the same direction with"),
    ("connect", "adjacent"),
    ("connect", "intersect"),
    ("converge", "connect"),
    ("converge", "intersect"),
    ("parallelly parked on", "isolatedly parked on"),
    ("randomly parked on", "parallelly parked on"),
    ("incorrectly parked on", "parallelly parked on"),
    ("docked alongside with", "docking at the same dock with"),
]

_PROTO_INIT_STOPWORDS = {"in", "the", "with", "of", "at", "on", "to", "from"}
_PROTO_INIT_STRONG_MODIFIERS = {
    "same": 2.5,
    "different": 2.5,
    "opposite": 2.5,
    "not": 2.5,
    "directly": 2.5,
    "indirectly": 2.5,
    "safe": 2.5,
    "danger": 2.5,
}
_PROTO_INIT_STATE_MODIFIERS = {
    "parallelly": 1.8,
    "isolatedly": 1.8,
    "randomly": 1.8,
    "incorrectly": 1.8,
    "slightly": 1.8,
    "violently": 1.8,
}
_PROTO_INIT_TOKEN_ALIAS = {
    "co": ("together",),
    "alongside": ("side", "parallel"),
    "breakwater": ("harbor", "barrier"),
    "taxiway": ("runway",),
    "running": ("run",),
    "driving": ("drive",),
}
_PROTO_INIT_SIGN_MODIFIERS = {
    "same": 1.0,
    "different": -1.0,
    "safe": 1.0,
    "danger": -1.0,
    "directly": 1.0,
    "indirectly": -1.0,
    "parallelly": 1.0,
    "isolatedly": -1.0,
    "slightly": 1.0,
    "violently": -1.0,
    "approach": 1.0,
    "away": -1.0,
    "toward": 1.0,
    "off": -1.0,
}
_PROTO_INIT_NEGATE_MODIFIERS = {"not", "opposite"}


def _canonicalize_pair_names(pair_names: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for pair in pair_names:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid semantic pair entry: {pair!r}")
        left = str(pair[0]).strip()
        right = str(pair[1]).strip()
        if not left or not right:
            raise ValueError(f"Semantic pair names must be non-empty: {pair!r}")
        result.append((left, right))
    return result


def _resolve_relation_name_pairs(
    relation_names: Sequence[str],
    configured_pairs,
    default_pairs: Sequence[tuple[str, str]],
    pair_kind: str,
) -> list[tuple[int, int]]:
    resolved_names = _canonicalize_pair_names(configured_pairs) if configured_pairs else list(default_pairs)
    name_to_idx = {str(name): idx for idx, name in enumerate(relation_names)}
    resolved_ids: list[tuple[int, int]] = []
    missing: list[str] = []
    seen = set()
    for left, right in resolved_names:
        left_idx = name_to_idx.get(left)
        right_idx = name_to_idx.get(right)
        if left_idx is None:
            missing.append(left)
        if right_idx is None:
            missing.append(right)
        if left_idx is None or right_idx is None:
            continue
        key = tuple(sorted((int(left_idx), int(right_idx))))
        if key in seen:
            continue
        seen.add(key)
        resolved_ids.append((int(left_idx), int(right_idx)))
    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise ValueError(f"Unknown {pair_kind} predicate names: {unique_missing}")
    return resolved_ids


def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * max(num_layers - 1, 0)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if idx < self.num_layers - 1 else layer(x)
        return x


def _empty_edge_index(device: torch.device | None = None) -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def _proposal_len(proposal) -> int:
    return int(proposal if isinstance(proposal, int) else len(proposal))


def _proposal_debug_id(proposal, fallback_idx: int) -> int:
    if hasattr(proposal, "has_field") and proposal.has_field("image_id"):
        image_id = proposal.get_field("image_id")
        if torch.is_tensor(image_id) and image_id.numel() > 0:
            return int(image_id.reshape(-1)[0].item())
    return int(fallback_idx)


def _estimate_gcn_message_bytes(num_edges: int, feat_dim: int, dtype: torch.dtype) -> int:
    dtype_sizes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
    }
    bytes_per_value = dtype_sizes.get(dtype, 4)
    return int(num_edges) * int(feat_dim) * int(bytes_per_value)


def _format_num_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(max(num_bytes, 0))
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def _graph_debug_enabled(proposals: Sequence) -> bool:
    if not proposals:
        return False
    proposal = proposals[0]
    cfg = getattr(proposal, "_debug_cfg", None)
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("TEST", {}).get("GRAPH_DEBUG", {}).get("ENABLED", False))


def _build_edge_indices_cpu(proposals: Sequence, rel_pair_idxs) -> Dict[str, torch.Tensor]:
    graph_debug_enabled = _graph_debug_enabled(proposals)
    total_objs = sum(_proposal_len(proposal) for proposal in proposals)
    total_rels = sum(int(pair_idx.size(0)) for pair_idx in rel_pair_idxs)
    if total_objs == 0:
        empty = torch.zeros((0, 0), dtype=torch.float32)
        return {
            "subj_to_rel": empty,
            "obj_to_rel": empty,
            "rel_to_rel_subj": empty,
            "rel_to_rel_obj": empty,
            "rel_to_rel_chain": empty,
            "entity_to_entity": empty,
        }

    obj_offset = 0
    rel_inds = []
    per_image_debug = []
    for image_idx, (proposal, pair_idx) in enumerate(zip(proposals, rel_pair_idxs)):
        num_obj = _proposal_len(proposal)
        if pair_idx.numel() == 0:
            local_entity_map = torch.zeros((num_obj, num_obj), dtype=torch.bool)
            per_image_debug.append(
                {
                    "image_id": _proposal_debug_id(proposal, image_idx),
                    "num_obj": num_obj,
                    "num_rel": 0,
                    "same_subj": 0,
                    "same_obj": 0,
                    "chain": 0,
                    "entity": int(local_entity_map.sum().item()),
                }
            )
            obj_offset += num_obj
            continue
        pair_cpu = pair_idx.detach().to(device="cpu", dtype=torch.long).clone()
        local_subj = pair_cpu[:, 0].clone()
        local_obj = pair_cpu[:, 1].clone()
        pair_cpu += obj_offset
        rel_inds.append(pair_cpu)

        subj_counts = torch.bincount(local_subj, minlength=num_obj)
        obj_counts = torch.bincount(local_obj, minlength=num_obj)
        same_subj_edges = int((subj_counts * (subj_counts - 1)).sum().item())
        same_obj_edges = int((obj_counts * (obj_counts - 1)).sum().item())
        cross = torch.zeros((num_obj, num_obj), dtype=torch.bool)
        cross[local_obj, local_subj] = True
        chain_edges = int((cross.sum(dim=0) * cross.sum(dim=1)).sum().item())
        local_entity_map = cross | cross.t()
        local_entity_map.fill_diagonal_(False)
        per_image_debug.append(
            {
                "image_id": _proposal_debug_id(proposal, image_idx),
                "num_obj": num_obj,
                "num_rel": int(pair_cpu.size(0)),
                "same_subj": same_subj_edges,
                "same_obj": same_obj_edges,
                "chain": chain_edges,
                "entity": int(local_entity_map.sum().item()),
            }
        )

        obj_offset += num_obj

    if total_rels == 0 or not rel_inds:
        empty_obj_rel = torch.zeros((total_objs, 0), dtype=torch.float32)
        empty_rel_rel = torch.zeros((0, 0), dtype=torch.float32)
        empty_entity = torch.zeros((total_objs, total_objs), dtype=torch.float32)
        return {
            "subj_to_rel": empty_obj_rel,
            "obj_to_rel": empty_obj_rel.clone(),
            "rel_to_rel_subj": empty_rel_rel,
            "rel_to_rel_obj": empty_rel_rel.clone(),
            "rel_to_rel_chain": empty_rel_rel.clone(),
            "entity_to_entity": empty_entity,
        }

    rel_inds = torch.cat(rel_inds, dim=0)
    device = rel_inds.device
    dtype = torch.float32
    rel_ids = torch.arange(total_rels, device=device)

    subj_pred_map = torch.zeros((total_objs, total_rels), dtype=dtype, device=device)
    obj_pred_map = torch.zeros((total_objs, total_rels), dtype=dtype, device=device)
    subj_pred_map[rel_inds[:, 0], rel_ids] = 1.0
    obj_pred_map[rel_inds[:, 1], rel_ids] = 1.0

    pred_pred_map_same_subj = (subj_pred_map.t() @ subj_pred_map) > 0
    pred_pred_map_same_obj = (obj_pred_map.t() @ obj_pred_map) > 0
    pred_pred_map_chain = (obj_pred_map.t() @ subj_pred_map) > 0
    pred_pred_map_same_subj.fill_diagonal_(False)
    pred_pred_map_same_obj.fill_diagonal_(False)
    pred_pred_map_chain.fill_diagonal_(False)
    pred_pred_map_same_subj = pred_pred_map_same_subj.to(dtype=dtype)
    pred_pred_map_same_obj = pred_pred_map_same_obj.to(dtype=dtype)
    pred_pred_map_chain = pred_pred_map_chain.to(dtype=dtype)

    cross = (obj_pred_map @ subj_pred_map.t()) > 0
    entity_map = (cross | cross.t())
    entity_map.fill_diagonal_(False)
    entity_map = entity_map.to(dtype=dtype)

    total_rel_graph_edges = int(
        pred_pred_map_same_subj.sum().item()
        + pred_pred_map_same_obj.sum().item()
        + pred_pred_map_chain.sum().item()
    )
    if graph_debug_enabled and (total_rels >= 20000 or total_rel_graph_edges >= 2_000_000):
        print(
            "RelationGraphDebug BatchSummary:",
            {
                "images": len(per_image_debug),
                "total_objs": int(total_objs),
                "total_rels": int(total_rels),
                "rel_to_rel_subj": int(pred_pred_map_same_subj.sum().item()),
                "rel_to_rel_obj": int(pred_pred_map_same_obj.sum().item()),
                "rel_to_rel_chain": int(pred_pred_map_chain.sum().item()),
                "entity_to_entity": int(entity_map.sum().item()),
            },
            flush=True,
        )
        for item in sorted(
            per_image_debug,
            key=lambda row: (row["same_subj"] + row["same_obj"] + row["chain"], row["num_rel"]),
            reverse=True,
        )[:8]:
            print("RelationGraphDebug Image:", item, flush=True)
    return {
        "subj_to_rel": subj_pred_map,
        "obj_to_rel": obj_pred_map,
        "rel_to_rel_subj": pred_pred_map_same_subj,
        "rel_to_rel_obj": pred_pred_map_same_obj,
        "rel_to_rel_chain": pred_pred_map_chain,
        "entity_to_entity": entity_map,
    }


def _move_edge_indices_to_device(edge_indices: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in edge_indices.items()
    }


def _encode_box_info(proposals: Sequence) -> torch.Tensor:
    infos = []
    for proposal in proposals:
        boxes = _proposal_box_tensor(proposal)
        if boxes.numel() == 0:
            infos.append(boxes.new_zeros((0, 9)))
            continue
        width = max(float(proposal.size[0]), 1.0)
        height = max(float(proposal.size[1]), 1.0)
        if proposal.mode == "xywha":
            cx, cy, w, h, a = boxes.split(1, dim=1)
            img_diag = math.sqrt(width * width + height * height)
            box_diag = torch.sqrt(w.square() + h.square())
            infos.append(
                torch.cat(
                    [
                        w / width,
                        h / height,
                        cx / width,
                        cy / height,
                        w / h.clamp(min=1e-6),
                        a,
                        box_diag / max(img_diag, 1e-6),
                        (w * h) / (width * height),
                        cx / cy.clamp(min=1e-6),
                    ],
                    dim=1,
                )
            )
        else:
            if proposal.mode == "xywh":
                boxes = proposal.convert("xyxy").bbox
            wh = boxes[:, 2:] - boxes[:, :2] + 1.0
            ctr = boxes[:, :2] + 0.5 * wh
            w, h = wh.split(1, dim=1)
            x, y = ctr.split(1, dim=1)
            x1, y1, x2, y2 = boxes.split(1, dim=1)
            infos.append(
                torch.cat(
                    [
                        w / width,
                        h / height,
                        x / width,
                        y / height,
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                        (w * h) / (width * height),
                    ],
                    dim=1,
                )
            )
    if not infos:
        return torch.zeros((0, 9))
    return torch.cat(infos, dim=0)


def _obb_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((boxes.size(0), 4))
    center, w, h, theta = torch.split(boxes, [2, 1, 1, 1], dim=-1)
    theta = torch.deg2rad(theta)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w * 0.5 * cos_theta) + torch.abs(h * 0.5 * sin_theta)
    y_bias = torch.abs(w * 0.5 * sin_theta) + torch.abs(h * 0.5 * cos_theta)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)


def _get_box_info(proposal) -> torch.Tensor:
    boxes = _proposal_box_tensor(proposal)
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8))
    if proposal.mode == "xywha":
        scale = float(max(max(proposal.size[0], proposal.size[1]), 100))
        xyxy = _obb_to_xyxy(boxes)
        return torch.cat((xyxy, boxes[:, :4]), dim=1) / scale
    if proposal.mode == "xywh":
        boxes = proposal.convert("xyxy").bbox
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), dim=1)
    scale = float(max(max(proposal.size[0], proposal.size[1]), 100))
    return torch.cat((boxes, center_box), dim=1) / scale


def _get_box_pair_info(box1: torch.Tensor, box2: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "xywha":
        union_box = box1[:, :4].clone()
        union_box[:, 0] = torch.min(box1[:, 0], box2[:, 0])
        union_box[:, 1] = torch.min(box1[:, 1], box2[:, 1])
        union_box[:, 2] = torch.max(box1[:, 2], box2[:, 2])
        union_box[:, 3] = torch.max(box1[:, 3], box2[:, 3])
        union_wh = union_box[:, 2:] - union_box[:, :2] + 1.0
        union_info = torch.cat((union_box, union_box[:, :2] + 0.5 * union_wh, union_wh), dim=1)

        inter_box = box1[:, :4].clone()
        inter_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
        inter_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
        inter_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
        inter_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
        invalid = (inter_box[:, 2] < inter_box[:, 0]) | (inter_box[:, 3] < inter_box[:, 1])
        inter_wh = inter_box[:, 2:] - inter_box[:, :2] + 1.0
        inter_info = torch.cat((inter_box, inter_box[:, :2] + 0.5 * inter_wh, inter_wh), dim=1)
        if invalid.any():
            inter_info[invalid] = 0
        return torch.cat((box1, box2, union_info, inter_info), dim=1)

    union_box = box1[:, :4].clone()
    union_box[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    union_box[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    union_box[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    union_box[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_wh = union_box[:, 2:] - union_box[:, :2] + 1.0
    union_info = torch.cat((union_box, union_box[:, :2] + 0.5 * union_wh, union_wh), dim=1)

    inter_box = box1[:, :4].clone()
    inter_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    inter_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    inter_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    inter_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    invalid = (inter_box[:, 2] < inter_box[:, 0]) | (inter_box[:, 3] < inter_box[:, 1])
    inter_wh = inter_box[:, 2:] - inter_box[:, :2] + 1.0
    inter_info = torch.cat((inter_box, inter_box[:, :2] + 0.5 * inter_wh, inter_wh), dim=1)
    if invalid.any():
        inter_info[invalid] = 0
    return torch.cat((box1, box2, union_info, inter_info), dim=1)


class PairwiseFeatureExtractor(nn.Module):
    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.obj_dim = in_channels
        self.union_dim = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_POOLING_DIM", default=in_channels)
        )
        self.hidden_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_HIDDEN_DIM", default=in_channels))
        self.pooling_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_POOLING_DIM", default=in_channels))
        self.embed_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "EMBED_DIM", default=200))
        self.num_obj_classes = int(_cfg_get(cfg, "MODEL", "ROI_BOX_HEAD", "NUM_CLASSES", default=1))
        self.class_names = list(_cfg_get(cfg, "MODEL", "ROI_BOX_HEAD", "CLASS_NAMES", default=[]))
        if len(self.class_names) < self.num_obj_classes + 1:
            self.class_names = self.class_names + [
                f"class_{idx}" for idx in range(len(self.class_names), self.num_obj_classes + 1)
            ]
        self.rel_feature_type = _cfg_get(
            cfg,
            "MODEL",
            "ROI_RELATION_HEAD",
            "EDGE_FEATURES_REPRESENTATION",
            default="fusion",
        )
        self.use_gt_box = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_BOX", default=False))
        self.use_gt_object_label = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_OBJECT_LABEL", default=False)
        )
        self.word_embed_feats_on = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "WORD_EMBEDDING_FEATURES", default=True)
        )
        self.spatial_for_vision = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CAUSAL", "SPATIAL_FOR_VISION", default=True)
        )
        self.semantic_glove_path = str(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "SEMANTIC_GLOVE_PATH", default="")
        )
        self.proto_text_init_modifier_aware = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "PROTO_TEXT_INIT_MODIFIER_AWARE", default=True)
        )
        self.box_mode = _cfg_get(cfg, "MODEL", "BOX_MODE", default="hbb")
        if not self.word_embed_feats_on:
            self.embed_dim = 0
        self.pair_geom_dim = 32

        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )
        if self.word_embed_feats_on:
            self.obj_embed_on_prob_dist = nn.Embedding(self.num_obj_classes + 1, self.embed_dim)
            self.obj_embed_on_pred_label = nn.Embedding(self.num_obj_classes + 1, self.embed_dim)
            obj_embed_init, obj_embed_mask, _ = _build_semantic_prototypes(
                self.class_names,
                self.embed_dim,
                self.semantic_glove_path,
                self.embed_dim,
                "semantic",
                modifier_aware=self.proto_text_init_modifier_aware,
            )
            with torch.no_grad():
                self.obj_embed_on_prob_dist.weight.normal_(mean=0.0, std=0.02)
                self.obj_embed_on_pred_label.weight.normal_(mean=0.0, std=0.02)
                if obj_embed_mask.any():
                    self.obj_embed_on_prob_dist.weight[obj_embed_mask] = obj_embed_init[obj_embed_mask]
                    self.obj_embed_on_pred_label.weight[obj_embed_mask] = obj_embed_init[obj_embed_mask]
        else:
            self.obj_embed_on_prob_dist = None
            self.obj_embed_on_pred_label = None
        self.obj_hidden_linear = nn.Linear(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
        self.obj_feat_aug_finalize_fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.obj_dim + self.embed_dim, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        self.rel_feat_dim_not_match = self.pooling_dim != self.union_dim
        if self.rel_feat_dim_not_match:
            self.rel_feature_up_dim = nn.Linear(self.union_dim, self.pooling_dim)

        self.pairwise_obj_feat_updim_fc = nn.Linear(
            self.hidden_dim + self.obj_dim + self.embed_dim,
            self.hidden_dim * 2,
        )
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(
                nn.Linear(self.pair_geom_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(inplace=True),
            )
        else:
            self.spt_emb = None
        self.pairwise_rel_feat_finalize_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

    def _get_obj_labels(self, proposals: Sequence):
        if self.training or self.use_gt_box:
            return torch.cat(
                [
                    p.get_field("gt_labels") if p.has_field("gt_labels") else p.get_field("labels")
                    for p in proposals
                ],
                dim=0,
            )
        return None

    def _get_obj_embed_by_dist(self, proposals: Sequence, obj_labels: torch.Tensor | None):
        if not self.word_embed_feats_on:
            if proposals:
                device = proposals[0].bbox.device
            else:
                device = torch.device("cpu")
            return torch.zeros((sum(len(p) for p in proposals), 0), device=device)
        if self.use_gt_object_label and obj_labels is not None:
            return self.obj_embed_on_prob_dist(obj_labels.long().clamp(min=0, max=self.num_obj_classes))
        obj_logits = []
        for proposal in proposals:
            if proposal.has_field("predict_logits"):
                obj_logits.append(proposal.get_field("predict_logits"))
            else:
                labels = (
                    proposal.get_field("gt_labels") if proposal.has_field("gt_labels") else proposal.get_field("labels")
                ).long().clamp(min=0, max=self.num_obj_classes)
                obj_logits.append(F.one_hot(labels, num_classes=self.num_obj_classes + 1).float())
        if not obj_logits:
            weight = self.obj_embed_on_prob_dist.weight
            return weight.new_zeros((0, weight.size(1)))
        obj_logits = torch.cat(obj_logits, dim=0).detach()
        obj_prob = F.softmax(obj_logits, dim=1)
        return obj_prob @ self.obj_embed_on_prob_dist.weight

    def _get_obj_pred_labels(self, proposals: Sequence, obj_labels: torch.Tensor | None):
        if self.use_gt_object_label and obj_labels is not None:
            return obj_labels.long().clamp(min=0, max=self.num_obj_classes)
        labels = []
        for proposal in proposals:
            if proposal.has_field("pred_labels"):
                labels.append(proposal.get_field("pred_labels").long())
            elif proposal.has_field("labels"):
                labels.append(proposal.get_field("labels").long())
        if not labels:
            return None
        return torch.cat(labels, dim=0).clamp(min=0, max=self.num_obj_classes)

    def pairwise_rel_features(
        self,
        augment_obj_feat: torch.Tensor,
        union_features: torch.Tensor | None,
        rel_pair_idxs,
        proposals: Sequence,
    ) -> torch.Tensor:
        num_objs = [len(p) for p in proposals]
        fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat).view(-1, 2, self.hidden_dim)
        head_rep, tail_rep = fused[:, 0], fused[:, 1]
        head_split = head_rep.split(num_objs, dim=0) if num_objs else []
        tail_split = tail_rep.split(num_objs, dim=0) if num_objs else []

        obj_pair_feats = []
        pair_geo_feats = []
        for pair_idx, head_chunk, tail_chunk, proposal in zip(rel_pair_idxs, head_split, tail_split, proposals):
            if pair_idx.numel() == 0:
                continue
            box_info = _get_box_info(proposal)
            obj_pair_feats.append(
                torch.cat((head_chunk[pair_idx[:, 0]], tail_chunk[pair_idx[:, 1]]), dim=1)
            )
            pair_geo_feats.append(
                _get_box_pair_info(box_info[pair_idx[:, 0]], box_info[pair_idx[:, 1]], proposal.mode)
            )

        if not obj_pair_feats:
            device = augment_obj_feat.device
            return augment_obj_feat.new_zeros((0, self.pooling_dim), device=device)

        obj_pair_feats = torch.cat(obj_pair_feats, dim=0)
        pair_geo_feats = torch.cat(pair_geo_feats, dim=0)
        if self.spatial_for_vision:
            obj_pair_feats = obj_pair_feats * self.spt_emb(pair_geo_feats)
        rel_features = self.pairwise_rel_feat_finalize_fc(obj_pair_feats)
        return rel_features

    def forward(self, roi_features, union_features, proposals: Sequence, rel_pair_idxs):
        obj_labels = self._get_obj_labels(proposals)
        obj_embed_by_dist = self._get_obj_embed_by_dist(proposals, obj_labels)
        pos_embed = self.pos_embed(_encode_box_info(proposals))
        obj_pre_rep = torch.cat((roi_features, obj_embed_by_dist, pos_embed), dim=1)
        obj_hidden = self.obj_hidden_linear(obj_pre_rep)

        obj_pred_labels = self._get_obj_pred_labels(proposals, obj_labels)
        if not self.word_embed_feats_on:
            obj_embed_by_label = obj_embed_by_dist
        elif obj_pred_labels is None:
            obj_embed_by_label = obj_embed_by_dist
        else:
            obj_embed_by_label = self.obj_embed_on_pred_label(obj_pred_labels)
        augment_obj_feat = torch.cat((obj_embed_by_label, roi_features, obj_hidden), dim=1)

        if self.rel_feature_type in ("obj_pair", "fusion"):
            rel_features = self.pairwise_rel_features(augment_obj_feat, union_features, rel_pair_idxs, proposals)
            if self.rel_feature_type == "fusion" and union_features is not None:
                union_proj = self.rel_feature_up_dim(union_features) if self.rel_feat_dim_not_match else union_features
                rel_features = union_proj + rel_features
        elif self.rel_feature_type == "union":
            if union_features is None:
                rel_features = roi_features.new_zeros((0, self.pooling_dim))
            else:
                rel_features = self.rel_feature_up_dim(union_features) if self.rel_feat_dim_not_match else union_features
        else:
            raise ValueError(f"Unknown EDGE_FEATURES_REPRESENTATION: {self.rel_feature_type}")

        augment_obj_feat = self.obj_feat_aug_finalize_fc(augment_obj_feat)
        return augment_obj_feat, rel_features


def _normal_init_linear(module: nn.Linear, mean: float = 0.0, std: float = 0.01) -> None:
    with torch.no_grad():
        module.weight.normal_(mean, std)
        if module.bias is not None:
            module.bias.zero_()


def _normalize_dense_bipartite_map(map_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if map_tensor.numel() == 0:
        return map_tensor.to(dtype=dtype)
    norm_map = map_tensor.to(dtype=dtype)
    src_deg = norm_map.sum(dim=1, keepdim=True).clamp(min=1.0)
    dst_deg = norm_map.sum(dim=0, keepdim=True).clamp(min=1.0)
    norm_map = norm_map * src_deg.rsqrt()
    norm_map = norm_map * dst_deg.rsqrt()
    return norm_map


def _normalize_dense_adjacency(adj_map: torch.Tensor, dtype: torch.dtype, add_self_loops: bool = True) -> torch.Tensor:
    if adj_map.numel() == 0:
        return adj_map.to(dtype=dtype)
    norm_adj = adj_map.to(dtype=dtype)
    if add_self_loops:
        norm_adj = norm_adj + torch.eye(norm_adj.size(0), device=norm_adj.device, dtype=dtype)
    deg = norm_adj.sum(dim=1).clamp(min=1.0)
    inv_sqrt = deg.rsqrt()
    norm_adj = norm_adj * inv_sqrt[:, None]
    norm_adj = norm_adj * inv_sqrt[None, :]
    return norm_adj


class _PyGCollectionLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        _normal_init_linear(self.fc)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        map_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if map_tensor.numel() == 0 or target.size(0) == 0 or source.size(0) == 0:
            return target.new_zeros((target.size(0), self.fc.out_features))
        source_proj = F.relu(self.fc(source))
        norm_map = _normalize_dense_bipartite_map(map_tensor, source_proj.dtype)
        return norm_map.t().matmul(source_proj)


class _DenseGCNConvCompat(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        _normal_init_linear(self.lin)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, adj_map: torch.Tensor) -> torch.Tensor:
        support = self.lin(x)
        norm_adj = _normalize_dense_adjacency(adj_map, support.dtype, add_self_loops=True)
        out = norm_adj.matmul(support)
        if self.bias is not None:
            out = out + self.bias
        return out


class _PyGGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation=F.relu,
        bias: bool = True,
        residual: bool = True,
        dropout: float = 0.1,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv = _DenseGCNConvCompat(in_dim, out_dim, bias=bias)
        self.res_fc = nn.Linear(in_dim, out_dim, bias=False) if residual and in_dim != out_dim else None
        self.bn = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.debug_enabled = False

    def forward(self, x: torch.Tensor, adj_map: torch.Tensor) -> torch.Tensor:
        x_in = self.dropout(x)
        num_edges = int(adj_map.sum().item()) if adj_map.numel() > 0 else 0
        if self.debug_enabled and num_edges >= 2_000_000:
            print(
                "RelationGraphDebug GCNInput:",
                {
                    "nodes": int(x_in.size(0)),
                    "feat_dim": int(x_in.size(1)),
                    "edges": num_edges,
                    "dtype": str(x_in.dtype),
                    "estimated_dense_adjacency": _format_num_bytes(
                        int(adj_map.numel()) * {
                            torch.float16: 2,
                            torch.bfloat16: 2,
                            torch.float32: 4,
                            torch.float64: 8,
                        }.get(x_in.dtype, 4)
                    ),
                },
                flush=True,
            )
        try:
            out = self.conv(x_in, adj_map)
        except torch.cuda.OutOfMemoryError:
            if self.debug_enabled:
                print(
                    "RelationGraphDebug OOMBeforeGCN:",
                    {
                        "nodes": int(x_in.size(0)),
                        "feat_dim": int(x_in.size(1)),
                        "edges": num_edges,
                        "dtype": str(x_in.dtype),
                        "estimated_dense_adjacency": _format_num_bytes(
                            int(adj_map.numel()) * {
                                torch.float16: 2,
                                torch.bfloat16: 2,
                                torch.float32: 4,
                                torch.float64: 8,
                            }.get(x_in.dtype, 4)
                        ),
                    },
                    flush=True,
                )
            raise
        out = self.bn(out)
        if self.residual:
            out = out + (self.res_fc(x_in) if self.res_fc is not None else x_in)
        if self.activation is not None:
            out = self.activation(out)
        return out


class _PyGRelContextLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.rel_to_obj = _PyGCollectionLayer(dim, dim)
        self.obj_to_rel = _PyGCollectionLayer(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, rel_feat: torch.Tensor, obj_feat: torch.Tensor, obj_to_rel_map: torch.Tensor) -> torch.Tensor:
        if rel_feat.numel() == 0 or obj_to_rel_map.numel() == 0:
            return rel_feat
        rel_to_obj_map = obj_to_rel_map.t()
        obj_context = self.rel_to_obj(obj_feat, self.dropout(rel_feat), rel_to_obj_map)
        rel_context = self.obj_to_rel(rel_feat, obj_context, obj_to_rel_map)
        rel_next = self.norm(rel_feat + rel_context)
        rel_next = self.norm(rel_next + self.ffn(rel_next))
        return rel_next


class _PyGRelGraphLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.rel_gcn = _PyGGCNLayer(dim, dim, residual=True, dropout=dropout)

    def forward(self, rel_feat: torch.Tensor, adj_map: torch.Tensor) -> torch.Tensor:
        if rel_feat.numel() == 0 or adj_map.numel() == 0:
            return rel_feat
        return self.rel_gcn(rel_feat, adj_map)


class RelationMessagePassing(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ent2ent = _PyGGCNLayer(dim, dim, residual=True, dropout=dropout)
        self.subj_collect = _PyGCollectionLayer(dim, dim)
        self.obj_collect = _PyGCollectionLayer(dim, dim)
        self.rel_graph = _PyGRelGraphLayer(dim, dropout=dropout)
        self.debug_enabled = False

    def set_debug_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.debug_enabled = enabled
        self.ent2ent.debug_enabled = enabled
        self.rel_graph.rel_gcn.debug_enabled = enabled

    def forward(
        self,
        obj_feat: torch.Tensor,
        rel_feat: torch.Tensor,
        subj_to_rel_map: torch.Tensor,
        obj_to_rel_map: torch.Tensor,
        rel_to_rel_subj_map: torch.Tensor,
        rel_to_rel_obj_map: torch.Tensor,
        rel_to_rel_chain_map: torch.Tensor,
        entity_to_entity_map: torch.Tensor,
    ):
        obj_next = self.ent2ent(obj_feat, entity_to_entity_map)
        rel_sources = []
        if rel_feat.size(0) > 0:
            rel_sources.append(self.subj_collect(rel_feat, obj_feat, subj_to_rel_map))
            rel_sources.append(self.obj_collect(rel_feat, obj_feat, obj_to_rel_map))
            rel_sources.append(self.rel_graph(rel_feat, rel_to_rel_subj_map))
            rel_sources.append(self.rel_graph(rel_feat, rel_to_rel_obj_map))
            # rel_sources.append(self.rel_graph(rel_feat, rel_to_rel_chain_map))

        if rel_sources:
            rel_next = torch.stack(rel_sources, dim=0).mean(dim=0)
        else:
            rel_next = rel_feat
        return obj_next, rel_next


class PredicatePrototypeHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int,
        glove_init: torch.Tensor,
        tau: float = 16.0,
        use_proj: bool = True,
        use_bn: bool = True,
        ema_alpha: float | None = None,
        lambda_pull: float = 0.2,
        lambda_sep: float = 0.01,
        sep_type: str = "etf",
        gamma: float = 10.0,
        antonym_pairs: Sequence[tuple[int, int]] | None = None,
        competitor_pairs: Sequence[tuple[int, int]] | None = None,
        lambda_ant_sep: float = 0.05,
        lambda_comp_sep: float = 0.02,
        ant_sep_margin: float = -0.10,
        comp_sep_margin: float = 0.10,
    ):
        super().__init__()
        self.d = d
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=True)

        if use_proj:
            layers = [nn.Linear(d_in, d, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm1d(d))
            layers.extend([nn.ReLU(inplace=True), nn.Linear(d, d, bias=False)])
            self.proj = nn.Sequential(*layers)
        else:
            if d_in != d:
                raise ValueError("Prototype head without projection requires d_in == d")
            self.proj = nn.Identity()

        _, glove_dim = glove_init.shape
        base = glove_init.contiguous()
        base = base + 0.01 * torch.randn_like(base)
        self.base_prototypes = nn.Parameter(base)
        self.mapper = nn.Linear(glove_dim, d, bias=False)
        nn.init.orthogonal_(self.mapper.weight)

        self.ema_alpha = ema_alpha
        if ema_alpha is not None:
            with torch.no_grad():
                proto_init = l2n(self.mapper(self.base_prototypes), dim=1)
                self.register_buffer("proto_ema", proto_init.clone())

        self.lambda_pull = float(lambda_pull)
        self.lambda_sep = float(lambda_sep)
        self.sep_type = str(sep_type)
        self.gamma = float(gamma)
        self.lambda_ant_sep = float(lambda_ant_sep)
        self.lambda_comp_sep = float(lambda_comp_sep)
        self.ant_sep_margin = float(ant_sep_margin)
        self.comp_sep_margin = float(comp_sep_margin)
        self.antonym_pairs = [(int(a), int(b)) for a, b in (antonym_pairs or [])]
        self.competitor_pairs = [(int(a), int(b)) for a, b in (competitor_pairs or [])]

    def _current_glove_prototypes(self) -> torch.Tensor:
        proto = l2n(self.mapper(self.base_prototypes), dim=1)
        if self.ema_alpha is None or not hasattr(self, "proto_ema"):
            return proto
        return l2n(self.ema_alpha * proto + (1.0 - self.ema_alpha) * self.proto_ema, dim=1)

    @torch.no_grad()
    def update_glove_ema(self):
        if self.ema_alpha is None or not hasattr(self, "proto_ema"):
            return
        proto = l2n(self.mapper(self.base_prototypes).detach(), dim=1)
        self.proto_ema.copy_(l2n(self.ema_alpha * proto + (1.0 - self.ema_alpha) * self.proto_ema, dim=1))

    def _pair_margin_loss(
        self,
        prototypes: torch.Tensor,
        pair_ids: Sequence[tuple[int, int]],
        margin: float,
    ) -> torch.Tensor:
        if not pair_ids:
            return prototypes.sum() * 0.0
        sims = []
        for left, right in pair_ids:
            if left < 0 or right < 0 or left >= prototypes.size(0) or right >= prototypes.size(0):
                continue
            sims.append((prototypes[left] * prototypes[right]).sum())
        if not sims:
            return prototypes.sum() * 0.0
        sim_tensor = torch.stack(sims, dim=0)
        return F.relu(sim_tensor - margin).mean()

    def forward(self, feats: torch.Tensor, labels: torch.Tensor | None = None):
        if feats.numel() == 0:
            return feats.new_zeros((0, self.base_prototypes.size(0))), {}

        z = l2n(self.proj(feats), dim=1)
        proto = self._current_glove_prototypes()
        logits = self.tau * (z @ proto.t())

        losses = {}
        if not self.training:
            return logits, losses

        # self.update_glove_ema()
        if labels is not None and labels.numel() > 0:
            valid = (labels >= 0) & (labels < proto.size(0))
            if valid.any():
                valid_z = z[valid].detach()
                valid_labels = labels[valid].detach()
                proto_y = proto[valid_labels]
                losses["pull"] = self.lambda_pull * (1.0 - (valid_z * proto_y).sum(dim=1)).mean()
            else:
                losses["pull"] = z.sum() * 0.0

        if self.sep_type.lower() == "etf":
            num_classes = proto.size(0)
            if num_classes > 1:
                target = -1.0 / (num_classes - 1)
                gram = proto @ proto.t()
                mask = ~torch.eye(num_classes, dtype=torch.bool, device=proto.device)
                loss_sep = ((gram[mask] - target) ** 2).mean()
            else:
                loss_sep = proto.sum() * 0.0
        else:
            gram = proto @ proto.t()
            mask = ~torch.eye(proto.size(0), dtype=torch.bool, device=proto.device)
            loss_sep = torch.exp(self.gamma * gram[mask]).mean() if mask.any() else proto.sum() * 0.0
        losses["sep"] = self.lambda_sep * loss_sep
        losses["ant_sep"] = self.lambda_ant_sep * self._pair_margin_loss(
            proto,
            self.antonym_pairs,
            self.ant_sep_margin,
        )
        losses["comp_sep"] = self.lambda_comp_sep * self._pair_margin_loss(
            proto,
            self.competitor_pairs,
            self.comp_sep_margin,
        )

        return logits, losses


def _tokenize_phrase(phrase: str) -> list[str]:
    clean = phrase.replace("/", " ").replace("-", " ").replace("_", " ").lower()
    return [tok for tok in clean.split() if tok]


def _semantic_token_weight(token: str) -> float:
    if token in _PROTO_INIT_STRONG_MODIFIERS:
        return float(_PROTO_INIT_STRONG_MODIFIERS[token])
    if token in _PROTO_INIT_STATE_MODIFIERS:
        return float(_PROTO_INIT_STATE_MODIFIERS[token])
    return 1.0


def _collect_semantic_token_specs(
    phrase: str,
    glove: dict[str, torch.Tensor],
    modifier_aware: bool,
) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    tokens = _tokenize_phrase(phrase)
    specs: list[tuple[str, float]] = []
    missing_tokens: list[str] = []
    used_fallbacks: list[str] = []

    for token in tokens:
        if modifier_aware and token in _PROTO_INIT_STOPWORDS:
            continue
        if token in glove:
            specs.append((token, _semantic_token_weight(token) if modifier_aware else 1.0))
            continue
        aliases = _PROTO_INIT_TOKEN_ALIAS.get(token, ())
        alias_hits = [alias for alias in aliases if alias in glove]
        if alias_hits:
            used_fallbacks.append(f"{token}->{'/'.join(alias_hits)}")
            for alias in alias_hits:
                specs.append((alias, _semantic_token_weight(token) if modifier_aware else 1.0))
            continue
        missing_tokens.append(token)

    return specs, missing_tokens, used_fallbacks


def _build_weighted_phrase_vector(
    phrase: str,
    glove: dict[str, torch.Tensor],
    modifier_aware: bool,
) -> tuple[torch.Tensor | None, list[str], list[str]]:
    token_specs, missing_tokens, used_fallbacks = _collect_semantic_token_specs(phrase, glove, modifier_aware)
    if not token_specs:
        return None, missing_tokens, used_fallbacks

    content_vectors = []
    content_weights = []
    direction_scale = 1.0
    for token, weight in token_specs:
        if modifier_aware and token in _PROTO_INIT_SIGN_MODIFIERS:
            direction_scale *= float(_PROTO_INIT_SIGN_MODIFIERS[token])
            continue
        if modifier_aware and token in _PROTO_INIT_NEGATE_MODIFIERS:
            direction_scale *= -1.0
            continue
        content_vectors.append(glove[token])
        content_weights.append(weight)

    if not content_vectors:
        content_vectors = [glove[token] for token, _ in token_specs]
        content_weights = [weight for _, weight in token_specs]

    weight_tensor = torch.tensor(content_weights, dtype=torch.float32).unsqueeze(1)
    vec_tensor = torch.stack(content_vectors, dim=0)
    phrase_vec = (vec_tensor * weight_tensor).sum(dim=0) / weight_tensor.sum().clamp(min=1e-6)
    phrase_vec = direction_scale * phrase_vec
    return phrase_vec, missing_tokens, used_fallbacks


def _load_glove_vectors(glove_path: str, embed_dim: int) -> dict[str, torch.Tensor]:
    path = Path(glove_path)
    if not path.is_file():
        return {}
    cache_path = path.with_suffix(path.suffix + f".{embed_dim}.pt")
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if isinstance(cached, dict):
            return cached
    vectors: dict[str, torch.Tensor] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            vec = torch.tensor([float(v) for v in parts[1:]], dtype=torch.float32)
            vectors[word] = vec
    return vectors


def _project_text_vector(text_vec: torch.Tensor, feat_dim: int) -> torch.Tensor:
    if text_vec.numel() == feat_dim:
        return text_vec
    if text_vec.numel() > feat_dim:
        return text_vec[:feat_dim]
    repeats = (feat_dim + text_vec.numel() - 1) // text_vec.numel()
    return text_vec.repeat(repeats)[:feat_dim]


def _build_prototype_text_init(
    names: list[str],
    glove_path: str,
    embed_dim: int,
    mode: str,
    modifier_aware: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    init = torch.zeros((len(names), embed_dim), dtype=torch.float32)
    init_mode = str(mode).lower()
    if init_mode == "random":
        init.normal_(mean=0.0, std=0.02)
        return init, {"used_fallbacks": [], "missing_predicates": []}
    if init_mode != "semantic":
        return init, {"used_fallbacks": [], "missing_predicates": []}
    glove = _load_glove_vectors(glove_path, embed_dim)
    used_fallbacks: list[str] = []
    missing_predicates: list[dict[str, object]] = []
    for idx, name in enumerate(names):
        if idx == 0 and ("background" in name.lower() or name.startswith("__")):
            continue
        phrase_vec, missing_tokens, fallback_hits = _build_weighted_phrase_vector(name, glove, modifier_aware)
        used_fallbacks.extend(fallback_hits)
        if phrase_vec is None:
            missing_predicates.append({"predicate": str(name), "missing_tokens": missing_tokens})
            continue
        init[idx] = phrase_vec
    diagnostics = {
        "used_fallbacks": sorted(set(used_fallbacks)),
        "missing_predicates": missing_predicates,
    }
    return init, diagnostics


def _build_semantic_prototypes(
    names: list[str],
    feat_dim: int,
    glove_path: str,
    embed_dim: int,
    mode: str,
    modifier_aware: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    prototypes = torch.zeros((len(names), feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(names),), dtype=torch.bool)
    init_mode = mode.lower()
    if init_mode == "random":
        prototypes.normal_(mean=0.0, std=0.02)
        mask[:] = True
        return prototypes, mask, {"used_fallbacks": [], "missing_predicates": []}
    if init_mode != "semantic":
        return prototypes, mask, {"used_fallbacks": [], "missing_predicates": []}
    glove = _load_glove_vectors(glove_path, embed_dim)
    used_fallbacks: list[str] = []
    missing_predicates: list[dict[str, object]] = []
    for idx, name in enumerate(names):
        if idx == 0 and ("background" in name.lower() or name.startswith("__")):
            continue
        phrase_vec, missing_tokens, fallback_hits = _build_weighted_phrase_vector(name, glove, modifier_aware)
        used_fallbacks.extend(fallback_hits)
        if phrase_vec is None:
            missing_predicates.append({"predicate": str(name), "missing_tokens": missing_tokens})
            continue
        prototypes[idx] = F.normalize(_project_text_vector(phrase_vec, feat_dim), dim=0)
        mask[idx] = True
    diagnostics = {
        "used_fallbacks": sorted(set(used_fallbacks)),
        "missing_predicates": missing_predicates,
    }
    return prototypes, mask, diagnostics


def _build_proto_head_args(cfg: dict, relation_names: Sequence[str]) -> tuple[dict, dict[str, object]]:
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    semantic_push_enabled = bool(rel_cfg.get("PROTO_SEMANTIC_PUSH_ENABLED", True))
    configured_ant_pairs = rel_cfg.get("PROTO_ANTONYM_PAIRS", [])
    configured_comp_pairs = rel_cfg.get("PROTO_COMPETITOR_PAIRS", [])
    antonym_pairs = (
        _resolve_relation_name_pairs(
            relation_names,
            configured_ant_pairs,
            _DEFAULT_REL_ANTONYM_NAME_PAIRS,
            "antonym",
        )
        if semantic_push_enabled
        else []
    )
    competitor_pairs = (
        _resolve_relation_name_pairs(
            relation_names,
            configured_comp_pairs,
            _DEFAULT_REL_COMPETITOR_NAME_PAIRS,
            "competitor",
        )
        if semantic_push_enabled
        else []
    )
    args = {
        "lambda_pull": float(rel_cfg.get("PROTO_LAMBDA_PULL", 0.2)),
        "lambda_sep": float(rel_cfg.get("PROTO_LAMBDA_SEP", 0.002)),
        "lambda_ant_sep": float(rel_cfg.get("PROTO_LAMBDA_ANT_SEP", 0.05)),
        "lambda_comp_sep": float(rel_cfg.get("PROTO_LAMBDA_COMP_SEP", 0.02)),
        "sep_type": str(rel_cfg.get("PROTO_SEP_TYPE", "etf")),
        "ant_sep_margin": float(rel_cfg.get("PROTO_ANT_SEP_MARGIN", -0.10)),
        "comp_sep_margin": float(rel_cfg.get("PROTO_COMP_SEP_MARGIN", 0.10)),
        "antonym_pairs": antonym_pairs,
        "competitor_pairs": competitor_pairs,
    }
    diagnostics = {
        "semantic_push_enabled": semantic_push_enabled,
        "antonym_pairs": [(relation_names[a], relation_names[b]) for a, b in antonym_pairs],
        "competitor_pairs": [(relation_names[a], relation_names[b]) for a, b in competitor_pairs],
        "lambda_pull": args["lambda_pull"],
        "lambda_sep": args["lambda_sep"],
        "lambda_ant_sep": args["lambda_ant_sep"],
        "lambda_comp_sep": args["lambda_comp_sep"],
        "ant_sep_margin": args["ant_sep_margin"],
        "comp_sep_margin": args["comp_sep_margin"],
        "sep_type": args["sep_type"],
    }
    return args, diagnostics


def _print_proto_semantic_diagnostics(tag: str, text_diag: dict[str, object], loss_diag: dict[str, object]) -> None:
    print(
        f"[{tag}] proto semantic config: "
        f"enabled={loss_diag['semantic_push_enabled']}, "
        f"lambda_pull={loss_diag['lambda_pull']}, "
        f"lambda_sep={loss_diag['lambda_sep']}, "
        f"lambda_ant_sep={loss_diag['lambda_ant_sep']}, "
        f"lambda_comp_sep={loss_diag['lambda_comp_sep']}, "
        f"ant_margin={loss_diag['ant_sep_margin']}, "
        f"comp_margin={loss_diag['comp_sep_margin']}, "
        f"sep_type={loss_diag['sep_type']}",
        flush=True,
    )
    print(f"[{tag}] antonym pairs: {loss_diag['antonym_pairs']}", flush=True)
    print(f"[{tag}] competitor pairs: {loss_diag['competitor_pairs']}", flush=True)
    print(f"[{tag}] glove fallbacks: {text_diag.get('used_fallbacks', [])}", flush=True)
    print(f"[{tag}] predicates without semantic tokens: {text_diag.get('missing_predicates', [])}", flush=True)


class RPCM(nn.Module):
    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.num_rel_classes = int(cfg["MODEL"]["ROI_RELATION_HEAD"]["NUM_CLASSES"])
        self.hidden_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_HIDDEN_DIM", default=in_channels))
        self.pooling_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_POOLING_DIM", default=in_channels))
        self.mlp_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_MLP_DIM", default=1024))
        self.feat_update_step = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_FEAT_UPDATE_STEP", default=1))
        self.use_gt_box = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_BOX", default=False))
        self.use_gt_object_label = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_OBJECT_LABEL", default=False)
        )

        if self.use_gt_box:
            self.mode = "predcls" if self.use_gt_object_label else "sgcls"
        else:
            self.mode = "sgdet"

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        self.message_passing = nn.ModuleList(
            [RelationMessagePassing(self.pooling_dim) for _ in range(self.feat_update_step)]
        )
        self.use_proto = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_USE_PROTOTYPE", default=True))
        self.rel_proto_init_mode = str(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_PROTO_INIT", default="semantic")
        )
        self.obj_proto_init_mode = str(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_OBJ_PROTO_INIT", default=self.rel_proto_init_mode)
        )
        self.semantic_glove_path = str(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "SEMANTIC_GLOVE_PATH", default="")
        )
        self.proto_text_init_modifier_aware = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "PROTO_TEXT_INIT_MODIFIER_AWARE", default=True)
        )
        self.class_names = list(_cfg_get(cfg, "MODEL", "ROI_BOX_HEAD", "CLASS_NAMES", default=[]))
        if len(self.class_names) < self.num_obj_classes + 1:
            self.class_names = self.class_names + [
                f"class_{idx}" for idx in range(len(self.class_names), self.num_obj_classes + 1)
            ]
        self.relation_names = list(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RELATION_NAMES", default=[]))
        if len(self.relation_names) < self.num_rel_classes:
            self.relation_names = self.relation_names + [
                f"relation_{idx}" for idx in range(len(self.relation_names), self.num_rel_classes)
            ]
        dropout = float(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_DROPOUT", default=0.2))
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        self.rel_residual = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.rel_norm = nn.LayerNorm(self.mlp_dim)
        self.use_pair_label_prior = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_PAIR_LABEL_PRIOR", default=True)
        )
        self.pair_prior_dim = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_PAIR_PRIOR_DIM", default=128)
        )
        if self.use_pair_label_prior:
            self.obj_pair_prior_embed = nn.Embedding(self.num_obj_classes + 1, self.pair_prior_dim)
            nn.init.normal_(self.obj_pair_prior_embed.weight, mean=0.0, std=0.02)
            self.pair_prior_mlp = nn.Sequential(
                nn.Linear(self.pair_prior_dim * 3, self.mlp_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.mlp_dim, self.mlp_dim),
            )
            self.pair_prior_gate = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
            self.pair_prior_norm = nn.LayerNorm(self.mlp_dim)
        else:
            self.obj_pair_prior_embed = None
            self.pair_prior_mlp = None
            self.pair_prior_gate = None
            self.pair_prior_norm = None
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_dim, self.num_rel_classes),
        )
        rel_glove_init, rel_text_diag = _build_prototype_text_init(
            self.relation_names,
            self.semantic_glove_path,
            int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "EMBED_DIM", default=200)),
            self.rel_proto_init_mode,
            modifier_aware=self.proto_text_init_modifier_aware,
        )
        proto_head_args, proto_loss_diag = _build_proto_head_args(cfg, self.relation_names)
        self.proto_head = PredicatePrototypeHead(
            self.mlp_dim,
            self.mlp_dim,
            rel_glove_init,
            ema_alpha=float(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RPCM_PROTO_MOMENTUM", default=0.9)),
            **proto_head_args,
        )
        self.graph_debug_enabled = bool(_cfg_get(cfg, "TEST", "GRAPH_DEBUG", "ENABLED", default=False))
        for layer in self.message_passing:
            layer.set_debug_enabled(self.graph_debug_enabled)
        _print_proto_semantic_diagnostics("RPCM", rel_text_diag, proto_loss_diag)

    def _build_edge_indices(self, proposals: Sequence, rel_pair_idxs, device: torch.device):
        for proposal in proposals:
            setattr(proposal, "_debug_cfg", self.cfg)
        return _move_edge_indices_to_device(_build_edge_indices_cpu(proposals, rel_pair_idxs), device)

    def _get_object_labels(self, proposals: Sequence, device: torch.device) -> torch.Tensor:
        labels = []
        for proposal in proposals:
            if proposal.has_field("pred_labels"):
                obj_label = proposal.get_field("pred_labels").long()
            elif proposal.has_field("labels"):
                obj_label = proposal.get_field("labels").long()
            elif proposal.has_field("predict_logits"):
                obj_label = proposal.get_field("predict_logits").argmax(dim=1).long()
            else:
                obj_label = torch.zeros((len(proposal),), dtype=torch.long, device=device)
            labels.append(obj_label.to(device=device).clamp(min=0, max=self.num_obj_classes))
        if not labels:
            return torch.zeros((0,), dtype=torch.long, device=device)
        return torch.cat(labels, dim=0)

    def _build_pair_label_prior(self, proposals: Sequence, rel_pair_idxs, device: torch.device) -> torch.Tensor:
        if not self.use_pair_label_prior or self.obj_pair_prior_embed is None:
            return torch.zeros((0, self.mlp_dim), device=device)
        obj_labels = self._get_object_labels(proposals, device)
        if obj_labels.numel() == 0:
            return torch.zeros((0, self.mlp_dim), device=device)

        pair_indices = []
        obj_offset = 0
        for proposal, pair_idx in zip(proposals, rel_pair_idxs):
            if pair_idx.numel() > 0:
                pair_indices.append(pair_idx.to(device=device, dtype=torch.long) + obj_offset)
            obj_offset += len(proposal)
        if not pair_indices:
            return torch.zeros((0, self.mlp_dim), device=device)

        flat_pair_idx = torch.cat(pair_indices, dim=0)
        subj_labels = obj_labels[flat_pair_idx[:, 0]]
        obj_labels_pair = obj_labels[flat_pair_idx[:, 1]]
        subj_embed = self.obj_pair_prior_embed(subj_labels)
        obj_embed = self.obj_pair_prior_embed(obj_labels_pair)
        pair_prior = torch.cat([subj_embed, obj_embed, subj_embed * obj_embed], dim=1)
        return self.pair_prior_mlp(pair_prior)

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        del rel_binarys, logger
        obj_features, rel_features = self.pairwise_feature_extractor(
            roi_features,
            union_features,
            proposals,
            rel_pair_idxs,
        )

        edge_indices = self._build_edge_indices(proposals, rel_pair_idxs, obj_features.device)
        obj_feature_steps = [obj_features]
        rel_feature_steps = [rel_features]
        for layer in self.message_passing:
            if rel_features.numel() == 0:
                break
            next_obj_features, next_rel_features = layer(
                obj_features,
                rel_features,
                edge_indices["subj_to_rel"],
                edge_indices["obj_to_rel"],
                edge_indices["rel_to_rel_subj"],
                edge_indices["rel_to_rel_obj"],
                edge_indices["rel_to_rel_chain"],
                edge_indices["entity_to_entity"],
            )
            obj_features = next_obj_features
            rel_features = next_rel_features
            obj_feature_steps.append(obj_features)
            rel_feature_steps.append(rel_features)
        obj_features = torch.stack(obj_feature_steps, dim=0).mean(dim=0)
        rel_features = torch.stack(rel_feature_steps, dim=0).mean(dim=0)
        # obj_features = obj_feature_steps[-1]
        # rel_features = rel_feature_steps[-1]

        rel_hidden = self.down_samp(rel_features)
        rel_hidden = self.rel_norm(self.rel_residual(rel_hidden) + rel_hidden)
        if self.use_pair_label_prior and rel_hidden.numel() > 0:
            pair_prior = self._build_pair_label_prior(proposals, rel_pair_idxs, rel_hidden.device)
            if pair_prior.shape == rel_hidden.shape:
                prior_gate = torch.sigmoid(self.pair_prior_gate(torch.cat([rel_hidden, pair_prior], dim=1)))
                rel_hidden = self.pair_prior_norm(rel_hidden + prior_gate * pair_prior)
        proto_losses = {}
        if self.use_proto:
            flat_rel_labels = (
                torch.cat(rel_labels, dim=0)
                if (self.training and rel_labels is not None and rel_labels)
                else None
            )
            relation_logits, proto_losses = self.proto_head(rel_hidden, flat_rel_labels)
        else:
            relation_logits = self.rel_classifier(rel_hidden)

        num_rels = [pairs.size(0) for pairs in rel_pair_idxs]
        relation_logits = list(relation_logits.split(num_rels, dim=0)) if num_rels else []

        refine_logits = [proposal.get_field("predict_logits") for proposal in proposals if proposal.has_field("predict_logits")]
        if len(refine_logits) != len(proposals):
            refine_logits = [
                proposal.get_field("predict_logits")
                if proposal.has_field("predict_logits")
                else obj_features.new_zeros((len(proposal), 0))
                for proposal in proposals
            ]
        return relation_logits, refine_logits, dict(proto_losses)


class QueryHierarchyRelationPredictor(nn.Module):
    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.num_rel_classes = int(cfg["MODEL"]["ROI_RELATION_HEAD"]["NUM_CLASSES"])
        self.pooling_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_POOLING_DIM", default=in_channels))
        self.hidden_dim = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "CONTEXT_HIDDEN_DIM", default=in_channels))
        self.num_subgraphs = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "NUM_SUBGRAPH_QUERIES", default=8))
        self.num_heads = int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "SUBGRAPH_ATTENTION_HEADS", default=4))
        self.dropout = float(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "SUBGRAPH_DROPOUT", default=0.1))
        self.use_gt_box = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_BOX", default=False))
        self.use_gt_object_label = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_GT_OBJECT_LABEL", default=False)
        )
        self.use_proto = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "HIER_USE_PROTOTYPE", default=True))
        self.use_pair_scale = bool(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "USE_PAIR_SCALE_ENCODER", default=True))
        self.test_subgraph_completion_topk = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "TEST_SUBGRAPH_COMPLETION_TOPK", default=0)
        )
        self.test_base_pair_keep_topk = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "TEST_BASE_PAIR_KEEP_TOPK", default=0)
        )
        self.test_pairness_topk = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "TEST_PAIRNESS_TOPK", default=0)
        )
        self.test_pairness_completion_topk = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "TEST_PAIRNESS_COMPLETION_TOPK", default=0)
        )
        self.test_pairness_chunk_size = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "TEST_PAIRNESS_CHUNK_SIZE", default=8192)
        )
        self.pairness_loss_weight = float(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "PAIRNESS_LOSS_WEIGHT", default=0.2)
        )
        self.max_test_pairs_per_image = int(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "MAX_TEST_PAIRS_PER_IMAGE", default=10000)
        )

        if self.use_gt_box:
            self.mode = "predcls" if self.use_gt_object_label else "sgcls"
        else:
            self.mode = "sgdet"

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        self.subgraph_queries = nn.Parameter(torch.randn(self.num_subgraphs, self.pooling_dim) * 0.02)
        self.query_to_node_attn = nn.MultiheadAttention(
            embed_dim=self.pooling_dim,
            num_heads=max(1, self.num_heads),
            dropout=self.dropout,
            batch_first=True,
        )
        self.subgraph_self_attn = nn.MultiheadAttention(
            embed_dim=self.pooling_dim,
            num_heads=max(1, self.num_heads),
            dropout=self.dropout,
            batch_first=True,
        )
        self.subgraph_ffn = nn.Sequential(
            nn.Linear(self.pooling_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.pooling_dim),
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(self.pooling_dim * 2, self.pooling_dim),
            nn.ReLU(inplace=True),
        )
        self.local_relation_mlp = nn.Sequential(
            nn.Linear(self.pooling_dim * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.pooling_dim),
            nn.ReLU(inplace=True),
        )
        self.global_completion_mlp = nn.Sequential(
            nn.Linear(self.pooling_dim * 5, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.pooling_dim),
            nn.ReLU(inplace=True),
        )
        self.pair_scale_encoder = PairScaleEncoder(self.pooling_dim, self.hidden_dim)
        self.graph_message_passing = RelationMessagePassing(self.pooling_dim)
        self.pairness_classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_rel_classes),
        )
        self.relation_names = list(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "RELATION_NAMES", default=[]))
        if len(self.relation_names) < self.num_rel_classes:
            self.relation_names = self.relation_names + [
                f"relation_{idx}" for idx in range(len(self.relation_names), self.num_rel_classes)
            ]
        self.semantic_glove_path = str(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "SEMANTIC_GLOVE_PATH", default="")
        )
        self.proto_text_init_modifier_aware = bool(
            _cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "PROTO_TEXT_INIT_MODIFIER_AWARE", default=True)
        )
        rel_glove_init, rel_text_diag = _build_prototype_text_init(
            self.relation_names,
            self.semantic_glove_path,
            int(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "EMBED_DIM", default=200)),
            "semantic",
            modifier_aware=self.proto_text_init_modifier_aware,
        )
        proto_head_args, proto_loss_diag = _build_proto_head_args(cfg, self.relation_names)
        self.proto_head = PredicatePrototypeHead(
            self.pooling_dim,
            self.pooling_dim,
            rel_glove_init,
            ema_alpha=float(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "HIER_PROTO_MOMENTUM", default=0.9)),
            **proto_head_args,
        )
        self.graph_debug_enabled = bool(_cfg_get(cfg, "TEST", "GRAPH_DEBUG", "ENABLED", default=False))
        self.graph_message_passing.set_debug_enabled(self.graph_debug_enabled)
        _print_proto_semantic_diagnostics("HIER", rel_text_diag, proto_loss_diag)

    @torch.no_grad()
    def complete_test_pairs(self, proposals, rel_pair_idxs, roi_features):
        if self.training or self.test_subgraph_completion_topk <= 0:
            return rel_pair_idxs
        if not proposals:
            return rel_pair_idxs

        empty_pair_idxs = [
            proposal.bbox.new_zeros((0, 2), dtype=torch.long)
            for proposal in proposals
        ]
        obj_features, _ = self.pairwise_feature_extractor(
            roi_features,
            None,
            proposals,
            empty_pair_idxs,
        )
        num_objs = [len(p) for p in proposals]
        obj_chunks = list(obj_features.split(num_objs, dim=0)) if num_objs else []
        completed = []
        for proposal, base_pairs, node_chunk in zip(proposals, rel_pair_idxs, obj_chunks):
            completed.append(self._complete_pairs_for_image(proposal, base_pairs, node_chunk))
        return completed

    @torch.no_grad()
    def _complete_pairs_for_image(self, proposal, base_pairs: torch.Tensor, node_chunk: torch.Tensor):
        n = len(proposal)
        device = proposal.bbox.device
        if n <= 1 or node_chunk.numel() == 0:
            return base_pairs

        query = self.subgraph_queries.unsqueeze(0)
        node_tokens = node_chunk.unsqueeze(0)
        subgraph_tokens, _ = self.query_to_node_attn(query, node_tokens, node_tokens)
        subgraph_tokens = subgraph_tokens + self.subgraph_ffn(subgraph_tokens)
        subgraph_tokens, _ = self.subgraph_self_attn(subgraph_tokens, subgraph_tokens, subgraph_tokens)
        subgraph_tokens = subgraph_tokens.squeeze(0)

        membership_logits = node_chunk @ subgraph_tokens.t() / (self.pooling_dim ** 0.5)
        membership = torch.softmax(membership_logits, dim=1)
        primary_subgraph = membership.argmax(dim=1)

        candidate_mask = torch.ones((n, n), dtype=torch.bool, device=device)
        candidate_mask.fill_diagonal_(False)

        inter_mask = primary_subgraph[:, None] != primary_subgraph[None, :]
        if (candidate_mask & inter_mask).any():
            candidate_mask = candidate_mask & inter_mask

        subgraph_sim = F.normalize(subgraph_tokens, dim=1) @ F.normalize(subgraph_tokens, dim=1).t()
        subgraph_sim = subgraph_sim.clamp(min=0.0) + 1.0
        pair_scores = membership @ subgraph_sim @ membership.t()

        if self.test_base_pair_keep_topk > 0 and base_pairs.numel() > 0:
            keep_budget = min(self.test_base_pair_keep_topk, int(base_pairs.size(0)))
            base_scores = pair_scores[base_pairs[:, 0], base_pairs[:, 1]]
            keep = base_scores.topk(keep_budget, largest=True, sorted=False).indices
            base_pairs = base_pairs[keep]
        proposal.add_field("pruned_rel_pair_idxs", base_pairs)

        if base_pairs.numel() > 0:
            candidate_mask[base_pairs[:, 0], base_pairs[:, 1]] = False
        pair_scores = pair_scores.masked_fill(~candidate_mask, float("-inf"))
        num_candidates = int(candidate_mask.sum().item())
        if num_candidates <= 0:
            return base_pairs

        extra_budget = self.test_subgraph_completion_topk
        if self.max_test_pairs_per_image > 0:
            extra_budget = min(extra_budget, max(self.max_test_pairs_per_image - int(base_pairs.size(0)), 0))
        extra_budget = min(extra_budget, num_candidates)
        if extra_budget <= 0:
            return base_pairs

        flat_idx = pair_scores.flatten().topk(extra_budget, largest=True, sorted=False).indices
        extra_pairs = torch.stack([flat_idx // n, flat_idx % n], dim=1).long()
        if base_pairs.numel() == 0:
            return extra_pairs
        return torch.cat([base_pairs, extra_pairs], dim=0)

    @torch.no_grad()
    def filter_test_pairs(self, proposals, rel_pair_idxs, roi_features):
        if self.training or self.test_pairness_topk <= 0:
            return rel_pair_idxs
        if not proposals:
            return rel_pair_idxs

        _, rel_features = self.pairwise_feature_extractor(
            roi_features,
            None,
            proposals,
            rel_pair_idxs,
        )
        if self.use_pair_scale:
            rel_features = rel_features + self.pair_scale_encoder(_encode_pair_scale_info(proposals, rel_pair_idxs))

        pairness_scores = self.pairness_classifier(rel_features).squeeze(1)
        rel_splits = [pairs.size(0) for pairs in rel_pair_idxs]
        score_chunks = list(pairness_scores.split(rel_splits, dim=0)) if rel_splits else []

        filtered = []
        for proposal, pair_idx, scores in zip(proposals, rel_pair_idxs, score_chunks):
            if pair_idx.numel() == 0 or pair_idx.size(0) <= self.test_pairness_topk:
                filtered.append(pair_idx)
                proposal.add_field("pruned_rel_pair_idxs", pair_idx)
                continue
            keep = scores.topk(self.test_pairness_topk, largest=True, sorted=False).indices
            kept_pair_idx = pair_idx[keep]
            filtered.append(kept_pair_idx)
            proposal.add_field("pruned_rel_pair_idxs", kept_pair_idx)
        if self.test_pairness_completion_topk <= 0:
            return filtered

        completed = []
        for proposal, keep_pairs, roi_chunk in zip(proposals, filtered, roi_features.split([len(p) for p in proposals], dim=0)):
            completed.append(self._complete_pairs_by_pairness(proposal, keep_pairs, roi_chunk))
        return completed

    @torch.no_grad()
    def _complete_pairs_by_pairness(self, proposal, keep_pairs: torch.Tensor, roi_chunk: torch.Tensor):
        n = len(proposal)
        device = proposal.bbox.device
        if n <= 1 or roi_chunk.numel() == 0:
            return keep_pairs

        total_budget = self.max_test_pairs_per_image if self.max_test_pairs_per_image > 0 else n * (n - 1)
        extra_budget = min(self.test_pairness_completion_topk, max(total_budget - int(keep_pairs.size(0)), 0))
        if extra_budget <= 0:
            return keep_pairs

        idx = torch.arange(n, device=device)
        subj, obj = torch.meshgrid(idx, idx, indexing="ij")
        all_pairs = torch.stack([subj.reshape(-1), obj.reshape(-1)], dim=1)
        all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
        if keep_pairs.numel() > 0:
            used = torch.zeros((n, n), dtype=torch.bool, device=device)
            used[keep_pairs[:, 0], keep_pairs[:, 1]] = True
            all_pairs = all_pairs[~used[all_pairs[:, 0], all_pairs[:, 1]]]
        if all_pairs.numel() == 0:
            return keep_pairs

        proposal_list = [proposal]
        best_scores = []
        best_pairs = []
        chunk_size = max(1, self.test_pairness_chunk_size)
        for start in range(0, all_pairs.size(0), chunk_size):
            pair_chunk = all_pairs[start : start + chunk_size]
            _, rel_features = self.pairwise_feature_extractor(
                roi_chunk,
                None,
                proposal_list,
                [pair_chunk],
            )
            if self.use_pair_scale:
                rel_features = rel_features + self.pair_scale_encoder(_encode_pair_scale_info(proposal_list, [pair_chunk]))
            scores = self.pairness_classifier(rel_features).squeeze(1)
            keep = scores.topk(min(extra_budget, scores.numel()), largest=True, sorted=False).indices
            best_scores.append(scores[keep])
            best_pairs.append(pair_chunk[keep])

        if not best_pairs:
            return keep_pairs
        best_scores = torch.cat(best_scores, dim=0)
        best_pairs = torch.cat(best_pairs, dim=0)
        keep = best_scores.topk(min(extra_budget, best_scores.numel()), largest=True, sorted=False).indices
        extra_pairs = best_pairs[keep]
        if keep_pairs.numel() == 0:
            return extra_pairs
        return torch.cat([keep_pairs, extra_pairs], dim=0)

    def _per_image_graph_reasoning(self, node_chunk: torch.Tensor, pair_chunk: torch.Tensor, pair_idx: torch.Tensor):
        if node_chunk.numel() == 0 or pair_idx.numel() == 0:
            return node_chunk, pair_chunk
        edge_indices = _move_edge_indices_to_device(
            _build_edge_indices_cpu([node_chunk.size(0)], [pair_idx]),
            node_chunk.device,
        )
        return self.graph_message_passing(
            node_chunk,
            pair_chunk,
            edge_indices["subj_to_rel"],
            edge_indices["obj_to_rel"],
            edge_indices["rel_to_rel_subj"],
            edge_indices["rel_to_rel_obj"],
            edge_indices["rel_to_rel_chain"],
            edge_indices["entity_to_entity"],
        )

    def _per_image_subgraph_reasoning(self, node_chunk: torch.Tensor, pair_chunk: torch.Tensor, pair_idx: torch.Tensor):
        if node_chunk.numel() == 0:
            return node_chunk, pair_chunk, {}

        query = self.subgraph_queries.unsqueeze(0)
        node_tokens = node_chunk.unsqueeze(0)
        subgraph_tokens, assign_weights = self.query_to_node_attn(query, node_tokens, node_tokens)
        subgraph_tokens = subgraph_tokens + self.subgraph_ffn(subgraph_tokens)
        subgraph_tokens, _ = self.subgraph_self_attn(subgraph_tokens, subgraph_tokens, subgraph_tokens)
        subgraph_tokens = subgraph_tokens.squeeze(0)
        assign_weights = assign_weights.squeeze(0)

        # [N, K], soft memberships of nodes to subgraphs
        node_membership_logits = node_chunk @ subgraph_tokens.t() / (self.pooling_dim ** 0.5)
        node_membership = torch.softmax(node_membership_logits, dim=1)
        node_ctx = node_membership @ subgraph_tokens
        node_out = self.node_fuse(torch.cat((node_chunk, node_ctx), dim=1))

        aux = {}
        if pair_idx.numel() == 0:
            aux["assign_entropy"] = -(assign_weights * (assign_weights.clamp(min=1e-8).log())).sum(dim=1).mean()
            gram = F.normalize(subgraph_tokens, dim=1) @ F.normalize(subgraph_tokens, dim=1).t()
            eye = torch.eye(gram.size(0), device=gram.device)
            aux["query_diversity"] = ((gram - eye) ** 2).mean()
            return node_out, pair_chunk, aux

        subj = node_out[pair_idx[:, 0]]
        obj = node_out[pair_idx[:, 1]]
        subj_mem = node_membership[pair_idx[:, 0]]
        obj_mem = node_membership[pair_idx[:, 1]]
        pair_subgraph_weight = 0.5 * (subj_mem + obj_mem)
        pair_subgraph_ctx = pair_subgraph_weight @ subgraph_tokens
        global_ctx = subgraph_tokens.mean(dim=0, keepdim=True).expand(pair_idx.size(0), -1)
        if pair_chunk.numel() == 0:
            pair_chunk = node_chunk.new_zeros((pair_idx.size(0), self.pooling_dim))
        local_rel = self.local_relation_mlp(
            torch.cat((subj, obj, pair_subgraph_ctx, torch.abs(subj - obj), pair_chunk), dim=1)
        )
        rel_out = self.global_completion_mlp(
            torch.cat((local_rel, subj, obj, pair_subgraph_ctx, global_ctx), dim=1)
        )

        aux["assign_entropy"] = -(assign_weights * (assign_weights.clamp(min=1e-8).log())).sum(dim=1).mean()
        gram = F.normalize(subgraph_tokens, dim=1) @ F.normalize(subgraph_tokens, dim=1).t()
        eye = torch.eye(gram.size(0), device=gram.device)
        aux["query_diversity"] = ((gram - eye) ** 2).mean()
        return node_out, rel_out, aux

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        del rel_binarys, logger
        obj_features, rel_features = self.pairwise_feature_extractor(
            roi_features,
            union_features,
            proposals,
            rel_pair_idxs,
        )
        if self.use_pair_scale:
            rel_features = rel_features + self.pair_scale_encoder(_encode_pair_scale_info(proposals, rel_pair_idxs))

        _, pairness_features = self.pairwise_feature_extractor(
            roi_features,
            None,
            proposals,
            rel_pair_idxs,
        )
        if self.use_pair_scale:
            pairness_features = pairness_features + self.pair_scale_encoder(_encode_pair_scale_info(proposals, rel_pair_idxs))
        pairness_logits = self.pairness_classifier(pairness_features).squeeze(1)

        num_objs = [len(p) for p in proposals]
        num_rels = [pairs.size(0) for pairs in rel_pair_idxs]
        obj_chunks = list(obj_features.split(num_objs, dim=0)) if num_objs else []
        rel_chunks = list(rel_features.split(num_rels, dim=0)) if num_rels else []

        refined_obj_chunks = []
        completed_rel_chunks = []
        for node_chunk, pair_chunk, pair_idx in zip(obj_chunks, rel_chunks, rel_pair_idxs):
            node_out, rel_out = self._per_image_graph_reasoning(node_chunk, pair_chunk, pair_idx)
            refined_obj_chunks.append(node_out)
            completed_rel_chunks.append(rel_out)

        obj_features = torch.cat(refined_obj_chunks, dim=0) if refined_obj_chunks else obj_features.new_zeros((0, self.pooling_dim))
        rel_features = torch.cat(completed_rel_chunks, dim=0) if completed_rel_chunks else obj_features.new_zeros((0, self.pooling_dim))

        proto_losses = {}
        if self.use_proto:
            flat_rel_labels = (
                torch.cat(rel_labels, dim=0)
                if (self.training and rel_labels is not None and rel_labels)
                else None
            )
            relation_logits, proto_losses = self.proto_head(rel_features, flat_rel_labels)
        else:
            relation_logits = self.rel_classifier(rel_features)

        if not self.training and pairness_logits.numel() > 0:
            pairness_scores = torch.sigmoid(pairness_logits)
            for proposal, scores in zip(proposals, pairness_scores.split(num_rels, dim=0)):
                proposal.add_field("pred_pairness_scores", scores)
        relation_logits = list(relation_logits.split(num_rels, dim=0)) if num_rels else []

        add_losses = dict(proto_losses)
        if self.training and rel_labels is not None and rel_labels and pairness_logits.numel() > 0:
            pairness_targets = torch.cat([(labels > 0).float() for labels in rel_labels], dim=0)
            add_losses["loss_pairness"] = (
                F.binary_cross_entropy_with_logits(pairness_logits, pairness_targets)
                * self.pairness_loss_weight
            )

        refine_logits = [
            proposal.get_field("predict_logits")
            if proposal.has_field("predict_logits")
            else obj_features.new_zeros((len(proposal), 0))
            for proposal in proposals
        ]
        return relation_logits, refine_logits, add_losses


def make_roi_relation_predictor(cfg: dict, in_channels: int):
    predictor_name = str(_cfg_get(cfg, "MODEL", "ROI_RELATION_HEAD", "PREDICTOR", default="Placeholder")).upper()
    if predictor_name in {"HIER_SUBGRAPH", "QUERY_HIERARCHY", "QHSG"}:
        return QueryHierarchyRelationPredictor(cfg, in_channels)
    if predictor_name == "RPCM":
        return RPCM(cfg, in_channels)
    else:
        raise ValueError(f"Unknown ROI relation predictor: {predictor_name}")
