from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import box_iou_rotated

from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


def _cfg_get(cfg: dict, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dim1: int, hidden_dim2: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, encoding_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PairProposalGenerator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.enabled = bool(rel_cfg.get("PPG_ENABLED", False))
        self.filter_method = str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper()
        self.threshold = int(rel_cfg.get("PPG_PAIR_THRESHOLD", 10000))
        self.topk = int(rel_cfg.get("PPG_TOPK", 10000))
        self.chunk_size = max(1, int(rel_cfg.get("PPG_CHUNK_SIZE", 65536)))
        self.encoding_dim = int(rel_cfg.get("PPG_ENCODING_DIM", 25))
        self.hidden_dim1 = int(rel_cfg.get("PPG_HIDDEN_DIM1", 50))
        self.hidden_dim2 = int(rel_cfg.get("PPG_HIDDEN_DIM2", 50))
        self.num_obj_classes = int(_cfg_get(cfg, "MODEL", "ROI_BOX_HEAD", "NUM_CLASSES", default=49)) - 1
        self.num_spatial_features = 7
        self.input_dim = self.num_obj_classes * 2 + self.num_spatial_features
        self.box_mode = str(_cfg_get(cfg, "MODEL", "BOX_MODE", default="hbb")).lower()
        default_weight = (
            rel_cfg.get("PPG_MODEL_PATH_OBB", "pretrained/STAR_OBB.pth")
            if self.box_mode == "obb"
            else rel_cfg.get("PPG_MODEL_PATH_HBB", "pretrained/STAR_HBB.pth")
        )
        self.model_path = Path(default_weight)

        self.model1 = _Autoencoder(self.input_dim, self.encoding_dim, self.hidden_dim1, self.hidden_dim2)
        self.model2 = _Autoencoder(self.input_dim, self.encoding_dim, self.hidden_dim1, self.hidden_dim2)
        self.criterion = nn.MSELoss(reduction="none")
        self.loaded = False
        self.model1.eval()
        self.model2.eval()
        for param in self.parameters():
            param.requires_grad_(False)
        self._load_weights()

    def _load_weights(self) -> None:
        if not self.enabled or self.filter_method != "PPG":
            return
        if not self.model_path.is_file():
            print(f"PPG weights not found: {self.model_path}", flush=True)
            return
        ckpt = torch.load(self.model_path, map_location="cpu")
        sd1 = ckpt.get("model_state_dict1")
        sd2 = ckpt.get("model_state_dict2")
        if sd1 is None or sd2 is None:
            print(f"PPG checkpoint missing model_state_dict1/2: {self.model_path}", flush=True)
            return
        try:
            self.model1.load_state_dict(sd1)
            self.model2.load_state_dict(sd2)
        except RuntimeError as exc:
            print(f"PPG checkpoint shape mismatch: {exc}", flush=True)
            return
        self.loaded = True
        self.model1.eval()
        self.model2.eval()

    def should_filter(self, pair_idx: torch.Tensor) -> bool:
        return (
            self.enabled
            and self.filter_method == "PPG"
            and self.loaded
            and pair_idx.numel() > 0
            and pair_idx.size(0) > self.threshold
        )

    def filter_pairs(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        if not self.should_filter(pair_idx):
            return pair_idx
        if proposal.mode == "xywha":
            return self._filter_oriented(proposal, pair_idx)
        return self._filter_horizontal(proposal, pair_idx)

    def _filter_oriented(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        obj_labels = proposal.get_field("labels").long()
        obj_boxes = proposal.bbox.to(dtype=torch.float32)
        img_size = proposal.size
        losses = []
        self.model1.to(pair_idx.device)
        self.model2.to(pair_idx.device)
        for start in range(0, pair_idx.size(0), self.chunk_size):
            chunk = pair_idx[start : start + self.chunk_size]
            head_boxes = obj_boxes[chunk[:, 0]]
            tail_boxes = obj_boxes[chunk[:, 1]]
            head_labels = obj_labels[chunk[:, 0]]
            tail_labels = obj_labels[chunk[:, 1]]
            head_polys = self._rotated_box_vertices(head_boxes)
            tail_polys = self._rotated_box_vertices(tail_boxes)
            pair_meta = torch.cat((head_polys, tail_polys, head_labels[:, None], tail_labels[:, None]), dim=1)
            features = self._build_oriented_features(head_boxes, tail_boxes, pair_meta, img_size)
            losses.append(self._compute_chunk_losses(features))
        keep = self._select_keep_indices(losses, pair_idx.device)
        return pair_idx[keep]

    def _filter_horizontal(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        obj_labels = proposal.get_field("labels").long()
        boxes = proposal.convert("xyxy").bbox.to(dtype=torch.float32)
        losses = []
        self.model1.to(pair_idx.device)
        self.model2.to(pair_idx.device)
        for start in range(0, pair_idx.size(0), self.chunk_size):
            chunk = pair_idx[start : start + self.chunk_size]
            head = boxes[chunk[:, 0]]
            tail = boxes[chunk[:, 1]]
            ious = self._bbox_iou_xyxy(head, tail)
            head_wh = head[:, 2:] - head[:, :2]
            tail_wh = tail[:, 2:] - tail[:, :2]
            head_area = head_wh[:, 0] * head_wh[:, 1]
            tail_area = tail_wh[:, 0] * tail_wh[:, 1]
            union_x1 = torch.minimum(head[:, 0], tail[:, 0])
            union_y1 = torch.minimum(head[:, 1], tail[:, 1])
            union_x2 = torch.maximum(head[:, 2], tail[:, 2])
            union_y2 = torch.maximum(head[:, 3], tail[:, 3])
            union_wh = torch.stack((union_x2 - union_x1, union_y2 - union_y1), dim=1)
            union_area = union_wh[:, 0] * union_wh[:, 1]
            head_ctr = 0.5 * (head[:, :2] + head[:, 2:])
            tail_ctr = 0.5 * (tail[:, :2] + tail[:, 2:])
            dist = torch.norm(tail_ctr - head_ctr, dim=1)
            head_diag = torch.norm(head_wh, dim=1)
            tail_diag = torch.norm(tail_wh, dim=1)
            spatial = torch.stack(
                (
                    ious,
                    head_diag / tail_diag,
                    dist / head_diag,
                    dist / tail_diag,
                    head_area / tail_area,
                    head_area / union_area,
                    tail_area / union_area,
                ),
                dim=1,
            )
            cls1 = self._one_hot_labels(obj_labels[chunk[:, 0]], pair_idx.device)
            cls2 = self._one_hot_labels(obj_labels[chunk[:, 1]], pair_idx.device)
            features = torch.cat((cls1, cls2, spatial), dim=1)
            losses.append(self._compute_chunk_losses(features))
        keep = self._select_keep_indices(losses, pair_idx.device)
        return pair_idx[keep]

    def _compute_chunk_losses(self, features: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            inputs = features.float()
            out1 = self.model1(inputs)
            out2 = self.model2(out1)
            loss1 = self.criterion(out1, inputs).mean(dim=1)
            loss2 = self.criterion(out2, inputs).mean(dim=1)
            losses = 0.5 * (loss1 + loss2)
        return losses.cpu()

    def _select_keep_indices(self, loss_chunks: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not loss_chunks:
            return torch.zeros((0,), dtype=torch.long, device=device)
        losses = torch.cat(loss_chunks, dim=0)
        keep_k = min(int(losses.numel()), self.topk)
        # Keep reconstruction-loss order so one max-k run can serve multiple
        # evaluation cutoffs without recomputing expensive OBB features.
        keep = losses.topk(keep_k, largest=False, sorted=True).indices
        return keep.to(device=device, dtype=torch.long)

    def _one_hot_labels(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        label_idx = (labels.long() - 1).clamp(min=0, max=self.num_obj_classes - 1)
        return torch.eye(self.num_obj_classes, device=device, dtype=torch.float32)[label_idx]

    def _build_oriented_features(
        self,
        head_boxes: torch.Tensor,
        tail_boxes: torch.Tensor,
        pair_meta: torch.Tensor,
        img_size,
    ) -> torch.Tensor:
        ious = box_iou_rotated(head_boxes, tail_boxes, aligned=True).to(dtype=torch.float32)
        head_area = head_boxes[:, 2] * head_boxes[:, 3]
        tail_area = tail_boxes[:, 2] * tail_boxes[:, 3]
        union_boxes = self._poly_to_obb(pair_meta[:, :16]).to(device=head_boxes.device, dtype=torch.float32)
        union_area = union_boxes[:, 2] * union_boxes[:, 3]
        cx1, cy1 = head_boxes[:, 0], head_boxes[:, 1]
        cx2, cy2 = tail_boxes[:, 0], tail_boxes[:, 1]
        dist = torch.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        head_diag = torch.sqrt(head_boxes[:, 2] ** 2 + head_boxes[:, 3] ** 2)
        tail_diag = torch.sqrt(tail_boxes[:, 2] ** 2 + tail_boxes[:, 3] ** 2)
        spatial = torch.stack(
            (
                ious,
                head_diag / tail_diag,
                dist / head_diag,
                dist / tail_diag,
                head_area / tail_area,
                head_area / union_area,
                tail_area / union_area,
            ),
            dim=1,
        )
        cls1 = self._one_hot_labels(pair_meta[:, 16].long(), head_boxes.device)
        cls2 = self._one_hot_labels(pair_meta[:, 17].long(), head_boxes.device)
        return torch.cat((cls1, cls2, spatial), dim=1)

    def _rotated_box_vertices(self, rboxes: torch.Tensor) -> torch.Tensor:
        if rboxes.numel() == 0:
            return rboxes.new_zeros((0, 8))
        x_ctr, y_ctr, width, height, angle = rboxes.t()
        tl_x, tl_y = -0.5 * width, -0.5 * height
        br_x, br_y = 0.5 * width, 0.5 * height
        rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y]).reshape(2, 4, -1).permute(2, 0, 1)
        sin_a, cos_a = torch.sin(angle), torch.cos(angle)
        rot = torch.stack([cos_a, -sin_a, sin_a, cos_a]).reshape(2, 2, -1).permute(2, 0, 1)
        polys = rot.matmul(rects).permute(2, 1, 0).reshape(-1, rboxes.size(0)).t()
        polys[:, ::2] += x_ctr[:, None]
        polys[:, 1::2] += y_ctr[:, None]
        return polys.contiguous()

    def _poly_to_obb(self, polys: torch.Tensor) -> torch.Tensor:
        polys_np = polys.detach().cpu().numpy().reshape((-1, 8, 2)).astype(np.float32)
        boxes = []
        for poly in polys_np:
            (x, y), (w, h), angle = cv2.minAreaRect(poly)
            angle = np.deg2rad(angle)
            if w < h:
                w, h = h, w
                angle += np.pi / 2
            while angle >= np.pi / 2:
                angle -= np.pi
            while angle < -np.pi / 2:
                angle += np.pi
            boxes.append((x, y, w, h, angle))
        return torch.tensor(boxes, dtype=torch.float32)

    def _bbox_iou_xyxy(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        inter_x1 = torch.maximum(box1[:, 0], box2[:, 0])
        inter_y1 = torch.maximum(box1[:, 1], box2[:, 1])
        inter_x2 = torch.minimum(box1[:, 2], box2[:, 2])
        inter_y2 = torch.minimum(box1[:, 3], box2[:, 3])
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h
        area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)).clamp(min=1e-6)
        area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)).clamp(min=1e-6)
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-6)
