from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import torch


class PairGraphBuilder:
    """Build a bounded, degree-constrained PPN graph for train and inference."""

    def __init__(self, cfg: dict, ppn_filter, semantic_filter):
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.cfg = cfg
        self.ppn = ppn_filter
        self.semantic = semantic_filter
        self.topk = int(rel_cfg.get("GRAPH_TOPK", 10000))
        self.partition_size = int(rel_cfg.get("GRAPH_PARTITION_SIZE", 2000))
        self.max_degree = int(rel_cfg.get("GRAPH_MAX_DEGREE", 48))
        self.relaxed_degree = int(rel_cfg.get("GRAPH_RELAXED_MAX_DEGREE", 64))
        self.gt_inject_full_epochs = int(rel_cfg.get("GRAPH_GT_INJECT_FULL_EPOCHS", 5))
        self.gt_inject_end_epoch = int(rel_cfg.get("GRAPH_GT_INJECT_END_EPOCH", 10))
        self.cache_path = Path(str(rel_cfg.get("GRAPH_CACHE_PATH", "")))
        self.cache_entries = {}
        self._load_cache()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_cache(self):
        if not str(self.cache_path) or not self.cache_path.is_file():
            return
        payload = torch.load(self.cache_path, map_location="cpu")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        expected_hash = self._sha256(self.ppn.model_path) if self.ppn.model_path.is_file() else ""
        compatible = (
            metadata.get("ppn_sha256") == expected_hash
            and int(metadata.get("topk", -1)) == self.topk
        )
        if not compatible:
            print(f"Ignoring stale pair graph cache: {self.cache_path}", flush=True)
            return
        self.cache_entries = payload.get("entries", {})
        print(f"Loaded pair graph cache {self.cache_path}: {len(self.cache_entries)} images", flush=True)

    def _allowed_label_pairs(self, device: torch.device):
        if not self.semantic.enabled or not self.semantic.loaded:
            return None
        prior = self.semantic.pair_prior.to(device=device)
        return prior.gt(0).any(dim=2)

    def _rank_pairs(self, proposal) -> torch.Tensor:
        if proposal.has_field("image_id"):
            image_id = str(int(proposal.get_field("image_id").reshape(-1)[0]))
            cached = self.cache_entries.get(image_id)
            if cached is not None:
                return cached.to(device=proposal.bbox.device, dtype=torch.long)
        if not self.ppn.loaded or self.ppn.model is None:
            raise RuntimeError("TypedHyperRPCM requires a loaded PPN checkpoint")
        model_device = next(self.ppn.model.parameters()).device
        if model_device != proposal.bbox.device:
            self.ppn.model.to(proposal.bbox.device)
        pairs, _ = self.ppn.model.topk_pairs(
            proposal,
            topk=self.topk,
            block_size=self.ppn.chunk_size,
            allowed_label_pairs=self._allowed_label_pairs(proposal.bbox.device),
        )
        return pairs

    @staticmethod
    def _degree_pass(
        ranked: torch.Tensor,
        accepted: torch.Tensor,
        out_degree: torch.Tensor,
        in_degree: torch.Tensor,
        cap: int,
        limit: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = accepted.tolist()
        selected_set = set(int(v) for v in selected)
        for row, (head, tail) in enumerate(ranked.tolist()):
            if len(selected) >= limit:
                break
            if row in selected_set:
                continue
            if out_degree[head] >= cap or in_degree[tail] >= cap:
                continue
            selected.append(row)
            selected_set.add(row)
            out_degree[head] += 1
            in_degree[tail] += 1
        return (
            torch.tensor(selected, dtype=torch.long, device=ranked.device),
            out_degree,
            in_degree,
        )

    def _degree_constrained(self, ranked: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if ranked.numel() == 0:
            return ranked
        out_degree = torch.zeros(num_nodes, dtype=torch.long)
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        accepted = torch.zeros((0,), dtype=torch.long, device=ranked.device)
        accepted, out_degree, in_degree = self._degree_pass(
            ranked, accepted, out_degree, in_degree, self.max_degree, self.topk
        )
        if accepted.numel() < min(self.topk, len(ranked)):
            accepted, _, _ = self._degree_pass(
                ranked, accepted, out_degree, in_degree, self.relaxed_degree, self.topk
            )
        return ranked[accepted]

    def _gt_injection_ratio(self, epoch: int) -> float:
        if epoch <= self.gt_inject_full_epochs:
            return 1.0
        if epoch >= self.gt_inject_end_epoch:
            return 0.0
        # The first decay epoch starts at 100%; the final epoch reaches zero.
        width = max(self.gt_inject_end_epoch - self.gt_inject_full_epochs - 1, 1)
        return float(self.gt_inject_end_epoch - epoch) / width

    def _inject_gt(self, pairs: torch.Tensor, target, epoch: int) -> torch.Tensor:
        if target is None or not target.has_field("relation_triplets"):
            return pairs
        ratio = self._gt_injection_ratio(epoch)
        if ratio <= 0:
            return pairs
        gt = target.get_field("relation_triplets")[:, :2].long().to(pairs.device)
        existing = {(int(h), int(t)) for h, t in pairs.tolist()}
        missing = []
        for head, tail in gt.tolist():
            if head == tail or (head, tail) in existing:
                continue
            # Stable pseudo-random schedule, independent of dataloader order.
            value = ((head * 73856093) ^ (tail * 19349663) ^ (epoch * 83492791)) & 0xFFFF
            if value / 65535.0 <= ratio:
                missing.append((head, tail))
                existing.add((head, tail))
        if missing:
            pairs = torch.cat((pairs, torch.tensor(missing, device=pairs.device, dtype=torch.long)))
        return pairs

    def partition(self, pairs: torch.Tensor) -> List[torch.Tensor]:
        if len(pairs) == 0:
            return [pairs]
        # Stable subject-major ordering keeps local outgoing neighborhoods together.
        max_node = int(pairs.max().item()) + 1
        order = (pairs[:, 0] * max_node + pairs[:, 1]).argsort()
        ordered = pairs[order]
        return [ordered[i:i + self.partition_size] for i in range(0, len(ordered), self.partition_size)]

    @staticmethod
    def labels_for_pairs(target, pairs: torch.Tensor) -> torch.Tensor:
        labels = torch.zeros((len(pairs),), dtype=torch.long, device=pairs.device)
        if target is None or not target.has_field("relation_triplets"):
            return labels
        relation_map = {}
        for head, tail, predicate in target.get_field("relation_triplets").long().tolist():
            if head != tail:
                relation_map[(int(head), int(tail))] = int(predicate)
        for row, pair in enumerate(pairs.tolist()):
            labels[row] = relation_map.get((int(pair[0]), int(pair[1])), 0)
        return labels

    def build(
        self,
        proposals: Sequence,
        targets=None,
        training: bool = False,
    ):
        epoch = int(self.cfg.get("_CURRENT_EPOCH", 0))
        output_pairs, output_labels, output_partitions = [], [], []
        targets = targets if targets is not None else [None] * len(proposals)
        for image_index, (proposal, target) in enumerate(zip(proposals, targets)):
            allowed_label_pairs = self._allowed_label_pairs(proposal.bbox.device)
            if allowed_label_pairs is not None:
                proposal.add_field("semantic_allowed_label_pairs", allowed_label_pairs.unsqueeze(0))
            ranked = self._rank_pairs(proposal)
            proposal.add_field("ppn_ranked_pair_idxs", ranked)
            pairs = self._degree_constrained(ranked, len(proposal))
            proposal.add_field("degree_capped_pair_idxs", pairs)
            image_id = image_index
            if proposal.has_field("image_id"):
                image_id = int(proposal.get_field("image_id").reshape(-1)[0])
            if training:
                pairs = self._inject_gt(pairs, target, epoch)
            partitions = self.partition(pairs)
            partition_ids = torch.cat([
                torch.full((len(part),), idx, dtype=torch.long, device=pairs.device)
                for idx, part in enumerate(partitions)
            ]) if partitions else torch.zeros((0,), dtype=torch.long, device=pairs.device)
            pairs = torch.cat(partitions, dim=0) if partitions else pairs
            if training and len(partitions) > 1:
                chosen = (epoch + image_id + image_index) % len(partitions)
                pairs = partitions[chosen]
                partition_ids = torch.zeros((len(pairs),), dtype=torch.long, device=pairs.device)
            proposal.add_field("pair_partition_ids", partition_ids)
            output_pairs.append(pairs)
            output_labels.append(self.labels_for_pairs(target, pairs) if training else None)
            output_partitions.append(partition_ids)
        return output_pairs, output_labels, output_partitions
