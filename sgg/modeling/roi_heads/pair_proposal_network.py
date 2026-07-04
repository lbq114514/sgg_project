from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from sgg.structures.boxes import BoxList


class PairProposalNetwork(nn.Module):
    """Detector-independent relation-pair scorer using only labels and boxes.

    The number of parameters is independent of the number of entities.  ``topk_pairs``
    enumerates pairs in blocks and maintains a running top-k, so it never materializes
    an ``N x N x D`` pair-feature tensor.
    """

    def __init__(
        self,
        num_obj_classes: int,
        label_dim: int = 32,
        box_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        glove_init: Optional[torch.Tensor] = None,
        anchor_class_ids: Optional[Sequence[int]] = None,
        anchor_dim: int = 32,
        num_predicates: int = 0,
    ):
        super().__init__()
        self.num_obj_classes = int(num_obj_classes)
        self.glove_dim = 0 if glove_init is None else int(glove_init.size(1))
        self.anchor_class_ids = tuple(int(v) for v in (anchor_class_ids or ()))
        self.anchor_dim = int(anchor_dim) if self.anchor_class_ids else 0
        self.num_predicates = int(num_predicates)
        if glove_init is not None:
            if glove_init.ndim != 2 or glove_init.size(0) != self.num_obj_classes:
                raise ValueError(
                    "glove_init must have shape "
                    f"({self.num_obj_classes}, glove_dim), got {tuple(glove_init.shape)}"
                )
            glove_init = glove_init.detach().float().clone()
            glove_init[0].zero_()
            self.glove_embedding = nn.Embedding.from_pretrained(
                glove_init, freeze=True, padding_idx=0
            )
        else:
            self.glove_embedding = None
        self.label_embedding = nn.Embedding(self.num_obj_classes, label_dim, padding_idx=0)
        self.box_encoder = nn.Sequential(
            nn.Linear(10, box_dim), nn.LayerNorm(box_dim), nn.GELU(),
            nn.Linear(box_dim, box_dim), nn.GELU(),
        )
        if self.anchor_dim > 0:
            self.anchor_encoder = nn.Sequential(
                nn.Linear(len(self.anchor_class_ids) * 6, self.anchor_dim),
                nn.LayerNorm(self.anchor_dim), nn.GELU(),
                nn.Linear(self.anchor_dim, self.anchor_dim), nn.GELU(),
            )
        else:
            self.anchor_encoder = None
        pair_dim = 2 * (self.glove_dim + label_dim + box_dim + self.anchor_dim) + 14
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.predicate_scorer = (
            nn.Sequential(
                nn.Linear(pair_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, self.num_predicates),
            )
            if self.num_predicates > 0 else None
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _box_features(proposal: BoxList) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = proposal.bbox.float()
        image_w, image_h = float(proposal.size[0]), float(proposal.size[1])
        scale = boxes.new_tensor([max(image_w, 1.0), max(image_h, 1.0)])
        if proposal.mode == "xywha":
            center = boxes[:, :2]
            size = boxes[:, 2:4].clamp(min=1e-4)
            angle = boxes[:, 4] * (math.pi / 180.0)
        else:
            xyxy = proposal.convert("xyxy").bbox.float()
            center = 0.5 * (xyxy[:, :2] + xyxy[:, 2:])
            size = (xyxy[:, 2:] - xyxy[:, :2]).clamp(min=1e-4)
            angle = boxes.new_zeros((len(proposal),))
        center_n = center / scale
        size_n = size / scale
        area = (size_n[:, 0] * size_n[:, 1]).clamp(min=1e-8)
        aspect = (size[:, 0] / size[:, 1]).clamp(min=1e-4, max=1e4)
        features = torch.stack(
            (center_n[:, 0], center_n[:, 1], size_n[:, 0], size_n[:, 1],
             size_n[:, 0].log(), size_n[:, 1].log(), area.log(), aspect.log(),
             angle.sin(), angle.cos()), dim=1,
        )
        return features, center_n, size_n

    def encode_entities(self, proposal: BoxList) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = proposal.get_field("labels").long().clamp(0, self.num_obj_classes - 1)
        box_features, centers, sizes = self._box_features(proposal)
        label_features = [self.label_embedding(labels)]
        if self.glove_embedding is not None:
            label_features.insert(0, self.glove_embedding(labels))
        entity_parts = [*label_features, self.box_encoder(box_features)]
        if self.anchor_encoder is not None:
            entity_parts.append(self.anchor_encoder(
                self._anchor_features(labels, centers, sizes)
            ))
        entity = torch.cat(entity_parts, dim=1)
        return entity, centers, sizes

    def _anchor_features(
        self, labels: torch.Tensor, centers: torch.Tensor, sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Nearest infrastructure instance per class: box, distance and containment."""
        outputs = []
        for class_id in self.anchor_class_ids:
            anchor_mask = labels == class_id
            if not anchor_mask.any():
                outputs.append(centers.new_zeros((len(labels), 6)))
                continue
            anchor_centers = centers[anchor_mask]
            anchor_sizes = sizes[anchor_mask]
            # Temporary shape N x number_of_anchors; infrastructure classes are sparse.
            distances = torch.cdist(centers, anchor_centers)
            nearest_distance, nearest_idx = distances.min(dim=1)
            nearest_center = anchor_centers[nearest_idx]
            nearest_size = anchor_sizes[nearest_idx]
            inside = (
                (centers - nearest_center).abs()
                <= nearest_size.clamp(min=1e-6) / 2
            ).all(dim=1).float()
            outputs.append(torch.cat((
                nearest_center, nearest_size,
                nearest_distance[:, None], inside[:, None],
            ), dim=1))
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _pair_geometry(
        head_center: torch.Tensor, head_size: torch.Tensor,
        tail_center: torch.Tensor, tail_size: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-6
        delta = tail_center - head_center
        hdiag = head_size.square().sum(1).sqrt().clamp(min=eps)
        tdiag = tail_size.square().sum(1).sqrt().clamp(min=eps)
        dist = delta.square().sum(1).sqrt()
        harea = head_size.prod(1).clamp(min=eps)
        tarea = tail_size.prod(1).clamp(min=eps)
        h_min, h_max = head_center - head_size / 2, head_center + head_size / 2
        t_min, t_max = tail_center - tail_size / 2, tail_center + tail_size / 2
        inter = (torch.minimum(h_max, t_max) - torch.maximum(h_min, t_min)).clamp(min=0).prod(1)
        iou = inter / (harea + tarea - inter).clamp(min=eps)
        return torch.stack((
            delta[:, 0], delta[:, 1], delta[:, 0] / head_size[:, 0].clamp(min=eps),
            delta[:, 1] / head_size[:, 1].clamp(min=eps),
            delta[:, 0] / tail_size[:, 0].clamp(min=eps),
            delta[:, 1] / tail_size[:, 1].clamp(min=eps), dist, dist / hdiag,
            dist / tdiag, (tail_size[:, 0] / head_size[:, 0]).log(),
            (tail_size[:, 1] / head_size[:, 1]).log(), (tarea / harea).log(),
            iou, inter.gt(0).float(),
        ), dim=1)

    def pair_outputs(
        self, proposal: BoxList, pair_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pair_idx.numel() == 0:
            empty = proposal.bbox.new_zeros((0,))
            predicates = (
                proposal.bbox.new_zeros((0, self.num_predicates))
                if self.predicate_scorer is not None else None
            )
            return empty, predicates
        encoded = self.encode_entities(proposal)
        return self.pair_outputs_from_encoded(encoded, pair_idx)

    def pair_outputs_from_encoded(
        self,
        encoded: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pair_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        entity, centers, sizes = encoded
        if pair_idx.numel() == 0:
            empty = entity.new_zeros((0,))
            predicates = (
                entity.new_zeros((0, self.num_predicates))
                if self.predicate_scorer is not None else None
            )
            return empty, predicates
        h, t = pair_idx[:, 0].long(), pair_idx[:, 1].long()
        geometry = self._pair_geometry(centers[h], sizes[h], centers[t], sizes[t])
        pair_features = torch.cat((entity[h], entity[t], geometry), dim=1)
        pairness = self.pair_scorer(pair_features).squeeze(1)
        predicates = self.predicate_scorer(pair_features) if self.predicate_scorer is not None else None
        return pairness, predicates

    def score_pairs(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        return self.pair_outputs(proposal, pair_idx)[0]

    @torch.no_grad()
    def topk_pairs(
        self,
        proposal: BoxList,
        topk: int,
        block_size: int = 65536,
        allowed_label_pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return global top-k directed pairs with bounded temporary memory."""
        n, device = len(proposal), proposal.bbox.device
        if n <= 1 or topk <= 0:
            return torch.zeros((0, 2), dtype=torch.long, device=device), proposal.bbox.new_zeros((0,))
        labels = proposal.get_field("labels").long().clamp(0, self.num_obj_classes - 1)
        entity, centers, sizes = self.encode_entities(proposal)
        keep_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
        keep_scores = proposal.bbox.new_zeros((0,))
        rows_per_block = max(1, int(block_size) // n)
        for row_start in range(0, n, rows_per_block):
            row_end = min(row_start + rows_per_block, n)
            heads = torch.arange(row_start, row_end, device=device)[:, None].expand(-1, n).reshape(-1)
            tails = torch.arange(n, device=device)[None, :].expand(row_end - row_start, -1).reshape(-1)
            valid = heads != tails
            if allowed_label_pairs is not None:
                allowed = allowed_label_pairs.to(device=device)
                valid &= allowed[labels[heads], labels[tails]]
            heads, tails = heads[valid], tails[valid]
            if heads.numel() == 0:
                continue
            geometry = self._pair_geometry(centers[heads], sizes[heads], centers[tails], sizes[tails])
            scores = self.pair_scorer(torch.cat((entity[heads], entity[tails], geometry), 1)).squeeze(1)
            pairs = torch.stack((heads, tails), 1)
            all_scores = torch.cat((keep_scores, scores))
            all_pairs = torch.cat((keep_pairs, pairs))
            k = min(int(topk), all_scores.numel())
            keep_scores, selected = all_scores.topk(k, sorted=False)
            keep_pairs = all_pairs[selected]
        order = keep_scores.argsort(descending=True)
        return keep_pairs[order], keep_scores[order]


class PairProposalNetworkFilter(nn.Module):
    """Inference adapter for using a trained PPN inside the relation head."""

    def __init__(self, cfg: dict):
        super().__init__()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.enabled = bool(rel_cfg.get("PPN_ENABLED", False))
        self.filter_method = str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper()
        self.model_path = Path(
            str(rel_cfg.get("PPN_MODEL_PATH", "outputs/star_pair_proposal_network/model_best.pth"))
        )
        self.threshold = int(rel_cfg.get("PPN_PAIR_THRESHOLD", 10000))
        self.topk = int(rel_cfg.get("PPN_TOPK", 10000))
        self.chunk_size = max(1, int(rel_cfg.get("PPN_CHUNK_SIZE", 200000)))
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.loaded = False
        # Keep the independently trained PPN outside the parent module state dict.
        # This preserves strict loading compatibility with existing SGG checkpoints.
        object.__setattr__(self, "model", None)
        self.checkpoint_metadata = {}
        self._load_weights()

    def _load_weights(self) -> None:
        if not self.enabled or self.filter_method != "PPN":
            return
        if not self.model_path.is_file():
            print(f"PPN checkpoint not found: {self.model_path}", flush=True)
            return
        try:
            model, checkpoint = load_pair_proposal_checkpoint(self.model_path, map_location="cpu")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"Failed to load PPN checkpoint {self.model_path}: {exc}", flush=True)
            return
        expected_classes = int(checkpoint.get("model_config", {}).get("num_obj_classes", 0))
        if expected_classes != self.num_obj_classes:
            print(
                f"PPN class count mismatch: checkpoint={expected_classes}, config={self.num_obj_classes}",
                flush=True,
            )
            return
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "model", model)
        self.checkpoint_metadata = {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
        }
        self.loaded = True
        print(
            f"Loaded PPN checkpoint from {self.model_path}: "
            f"epoch={checkpoint.get('epoch', 'unknown')}, topk={self.topk}",
            flush=True,
        )

    def should_filter(self, pair_idx: torch.Tensor) -> bool:
        return (
            self.enabled and self.filter_method == "PPN" and self.loaded
            and self.model is not None and pair_idx.size(0) > self.threshold
        )

    @torch.no_grad()
    def filter_pairs(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        if not self.should_filter(pair_idx):
            return pair_idx
        model_device = next(self.model.parameters()).device
        if model_device != proposal.bbox.device:
            self.model.to(proposal.bbox.device)
        encoded = self.model.encode_entities(proposal)
        keep_pairs = torch.zeros((0, 2), dtype=torch.long, device=pair_idx.device)
        keep_scores = proposal.bbox.new_zeros((0,))
        for start in range(0, pair_idx.size(0), self.chunk_size):
            pairs = pair_idx[start:start + self.chunk_size]
            scores = self.model.pair_outputs_from_encoded(encoded, pairs)[0]
            all_pairs = torch.cat((keep_pairs, pairs))
            all_scores = torch.cat((keep_scores, scores))
            k = min(self.topk, all_scores.numel())
            keep_scores, selected = all_scores.topk(k, sorted=False)
            keep_pairs = all_pairs[selected]
        return keep_pairs[keep_scores.argsort(descending=True)]


def load_glove_class_matrix(
    glove_path: str | Path,
    class_names: Sequence[str],
) -> Tuple[torch.Tensor, dict]:
    """Build a class matrix from the repository's serialized GloVe format.

    Multi-token names are split on underscores, whitespace and hyphens, then
    represented by the mean of all available token vectors.
    """
    payload = torch.load(Path(glove_path), map_location="cpu")
    if not (isinstance(payload, (tuple, list)) and len(payload) == 3):
        raise ValueError("Expected GloVe checkpoint format: (word_to_idx, vectors, dim)")
    word_to_idx, vectors, glove_dim = payload
    if not isinstance(word_to_idx, dict) or not torch.is_tensor(vectors):
        raise ValueError("Invalid GloVe vocabulary or vector matrix")
    glove_dim = int(glove_dim)
    if vectors.ndim != 2 or vectors.size(1) != glove_dim:
        raise ValueError("GloVe vector dimension does not match checkpoint metadata")

    matrix = torch.zeros((len(class_names), glove_dim), dtype=torch.float32)
    missing = {}
    for class_id, name in enumerate(class_names):
        if class_id == 0 or "background" in str(name).lower():
            continue
        tokens = [token for token in re.split(r"[_\s-]+", str(name).lower()) if token]
        token_vectors = [vectors[word_to_idx[token]].float() for token in tokens if token in word_to_idx]
        missing_tokens = [token for token in tokens if token not in word_to_idx]
        if token_vectors:
            matrix[class_id] = torch.stack(token_vectors).mean(dim=0)
        if missing_tokens or not token_vectors:
            missing[str(name)] = missing_tokens or tokens
    diagnostics = {
        "glove_dim": glove_dim,
        "covered_classes": len(class_names) - 1 - sum(1 for i in range(1, len(class_names)) if not matrix[i].any()),
        "missing_tokens": missing,
    }
    return matrix, diagnostics


def load_pair_proposal_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[PairProposalNetwork, dict]:
    """Restore a trained PPN without requiring the original GloVe file."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid PairProposalNetwork checkpoint")
    config = dict(checkpoint.get("model_config", {}))
    required = {"num_obj_classes", "label_dim", "box_dim", "hidden_dim", "dropout"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Checkpoint is missing model config keys: {missing}")
    state_dict = checkpoint["model_state_dict"]
    glove_init = state_dict.get("glove_embedding.weight")
    model = PairProposalNetwork(
        num_obj_classes=int(config["num_obj_classes"]),
        label_dim=int(config["label_dim"]),
        box_dim=int(config["box_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        glove_init=glove_init,
        anchor_class_ids=config.get("anchor_class_ids", ()),
        anchor_dim=int(config.get("anchor_dim", 32)),
        num_predicates=int(config.get("num_predicates", 0)),
    )
    model.load_state_dict(state_dict)
    return model, checkpoint
