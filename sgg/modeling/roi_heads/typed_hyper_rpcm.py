from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sgg.modeling.roi_heads.roi_relation_predictors import (
    MLP,
    PairwiseFeatureExtractor,
    PredicatePrototypeHead,
    _build_proto_head_args,
    _build_prototype_text_init,
    _cfg_get,
)


FAMILY_PREDICATES: Dict[int, Tuple[int, ...]] = {
    0: (0,),
    1: (3, 4, 16, 17, 18, 19, 26, 47, 49, 54),
    2: (8, 21, 28, 34, 39, 40, 51),
    3: (10, 22, 25, 32, 35, 38, 50),
    4: (1, 2, 5, 6, 7, 9, 12, 13, 14, 23, 24, 27, 31, 33, 36),
    5: (11, 15, 20, 29, 30, 37, 41, 46, 48),
    6: (42, 43, 44, 45, 52, 53, 55, 56, 57, 58),
}


def validate_family_mapping(num_predicates: int = 59) -> torch.Tensor:
    mapping = torch.full((num_predicates,), -1, dtype=torch.long)
    for family, predicates in FAMILY_PREDICATES.items():
        for predicate in predicates:
            if predicate < 0 or predicate >= num_predicates or mapping[predicate] >= 0:
                raise ValueError(f"Invalid or duplicate predicate in family mapping: {predicate}")
            mapping[predicate] = family
    missing = torch.nonzero(mapping < 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(f"Predicates missing from family mapping: {missing}")
    return mapping


def _mean_aggregate(source: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = source.new_zeros((size, source.size(1)))
    count = source.new_zeros((size, 1))
    if index.numel() == 0:
        return out
    out.index_add_(0, index, source)
    count.index_add_(0, index, source.new_ones((len(index), 1)))
    return out / count.clamp(min=1.0)


class SparseTypedMessageLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.entity_msg = nn.Linear(dim, dim)
        self.relation_from_entity = nn.Linear(dim * 2, dim)
        self.relation_context = nn.Linear(dim * 2, dim)
        self.entity_norm = nn.LayerNorm(dim)
        self.relation_norm = nn.LayerNorm(dim)
        self.entity_ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.relation_ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, entities: torch.Tensor, relations: torch.Tensor, pairs: torch.Tensor):
        if pairs.numel() == 0:
            return entities, relations
        head, tail = pairs[:, 0].long(), pairs[:, 1].long()
        entity_source = torch.cat((self.entity_msg(entities[tail]), self.entity_msg(entities[head])))
        entity_index = torch.cat((head, tail))
        entity_delta = _mean_aggregate(entity_source, entity_index, len(entities))
        entities = self.entity_norm(entities + entity_delta)
        entities = self.entity_norm(entities + self.entity_ffn(entities))

        endpoint = self.relation_from_entity(torch.cat((entities[head], entities[tail]), dim=1))
        rel_at_head = _mean_aggregate(relations, head, len(entities))[head]
        rel_at_tail = _mean_aggregate(relations, tail, len(entities))[tail]
        context = self.relation_context(torch.cat((rel_at_head, rel_at_tail), dim=1))
        relations = self.relation_norm(relations + endpoint + context)
        relations = self.relation_norm(relations + self.relation_ffn(relations))
        return entities, relations


class SparseHypergraphLayer(nn.Module):
    """Relation factors connected to subject/object and optional anchor entities."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.entity_to_factor = nn.Linear(dim, dim)
        self.factor_to_entity = nn.Linear(dim, dim)
        self.factor_gate = nn.Linear(dim * 2, 1)
        self.factor_norm = nn.LayerNorm(dim)
        self.entity_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, entities: torch.Tensor, factors: torch.Tensor, incidence: torch.Tensor):
        if incidence.numel() == 0:
            return entities, factors
        entity_idx, factor_idx = incidence[0].long(), incidence[1].long()
        entity_msg = self.entity_to_factor(entities[entity_idx])
        gates = torch.sigmoid(self.factor_gate(torch.cat((entity_msg, factors[factor_idx]), dim=1)))
        factor_delta = _mean_aggregate(entity_msg * gates, factor_idx, len(factors))
        factors = self.factor_norm(factors + self.dropout(factor_delta))
        node_msg = self.factor_to_entity(factors[factor_idx])
        entity_delta = _mean_aggregate(node_msg, entity_idx, len(entities))
        entities = self.entity_norm(entities + self.dropout(entity_delta))
        return entities, factors


class TypedHyperRPCM(nn.Module):
    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.cfg = cfg
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.num_rel_classes = int(rel_cfg["NUM_CLASSES"])
        self.pooling_dim = int(rel_cfg.get("CONTEXT_POOLING_DIM", in_channels))
        self.mlp_dim = int(rel_cfg.get("RPCM_MLP_DIM", 2048))
        self.graph_dim = int(rel_cfg.get("TYPED_GRAPH_HIDDEN_DIM", 512))
        self.num_sparse_layers = int(rel_cfg.get("TYPED_SPARSE_LAYERS", 2))
        self.hypergraph_enabled = bool(rel_cfg.get("TYPED_HYPERGRAPH_ENABLED", False))
        self.num_hyper_layers = int(rel_cfg.get("TYPED_HYPERGRAPH_LAYERS", 2))
        dropout = float(rel_cfg.get("RPCM_DROPOUT", 0.2))

        family_map = validate_family_mapping(self.num_rel_classes)
        self.register_buffer("predicate_to_family", family_map, persistent=True)
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        self.rel_residual = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.rel_norm = nn.LayerNorm(self.mlp_dim)
        self.obj_to_graph = nn.Linear(self.pooling_dim, self.graph_dim)
        self.rel_to_graph = nn.Linear(self.pooling_dim, self.graph_dim)
        self.graph_to_hidden = nn.Linear(self.graph_dim, self.mlp_dim)

        self.geometry_expert = nn.Sequential(nn.Linear(14, self.graph_dim), nn.GELU(), nn.Linear(self.graph_dim, self.graph_dim))
        self.anchor_expert = nn.Sequential(nn.Linear(4, self.graph_dim), nn.GELU(), nn.Linear(self.graph_dim, self.graph_dim))
        self.group_expert = nn.Sequential(nn.Linear(4, self.graph_dim), nn.GELU(), nn.Linear(self.graph_dim, self.graph_dim))
        self.motion_expert = nn.Sequential(nn.Linear(6, self.graph_dim), nn.GELU(), nn.Linear(self.graph_dim, self.graph_dim))
        self.network_label_embed = nn.Embedding(self.num_obj_classes + 1, 32)
        self.anchor_label_embed = nn.Embedding(self.num_obj_classes + 1, 16)
        self.anchor_assignment = nn.Sequential(
            nn.Linear(36, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.network_expert = nn.Sequential(nn.Linear(68, self.graph_dim), nn.GELU(), nn.Linear(self.graph_dim, self.graph_dim))
        self.expert_norm = nn.LayerNorm(self.graph_dim)
        self.expert_family_gate = nn.Linear(self.graph_dim, len(FAMILY_PREDICATES))

        self.sparse_layers = nn.ModuleList([
            SparseTypedMessageLayer(self.graph_dim, dropout) for _ in range(self.num_sparse_layers)
        ])
        self.hyper_layers = nn.ModuleList([
            SparseHypergraphLayer(self.graph_dim, dropout) for _ in range(self.num_hyper_layers)
        ])
        self.family_head = nn.Linear(self.mlp_dim, len(FAMILY_PREDICATES))
        self.fine_heads = nn.ModuleDict({
            str(family): nn.Linear(self.mlp_dim, len(predicates))
            for family, predicates in FAMILY_PREDICATES.items() if family != 0
        })

        self.family_loss_weight = float(rel_cfg.get("TYPED_FAMILY_LOSS_WEIGHT", 0.5))
        self.anchor_loss_weight = float(rel_cfg.get("TYPED_ANCHOR_LOSS_WEIGHT", 0.2))
        self.logic_loss_weight = float(rel_cfg.get("TYPED_LOGIC_LOSS_WEIGHT", 0.1))
        self.hierarchy_weight = float(rel_cfg.get("TYPED_HIERARCHY_LOGIT_WEIGHT", 1.0))
        self.proto_logit_weight = float(rel_cfg.get("TYPED_PROTO_LOGIT_WEIGHT", 0.0))
        self.anchor_topk = int(rel_cfg.get("TYPED_ANCHOR_TOPK", 4))
        self.class_names = list(rel_cfg.get("_OBJECT_CLASS_NAMES", cfg["MODEL"]["ROI_BOX_HEAD"].get("CLASS_NAMES", [])))
        anchor_names = rel_cfg.get("TYPED_ANCHOR_CLASSES", [
            "apron", "dock", "taxiway", "runway", "breakwater", "truck_parking",
            "car_parking", "bridge", "intersection", "lattice_tower", "substation", "genset",
        ])
        name_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.anchor_class_ids = tuple(name_to_id[name] for name in anchor_names if name in name_to_id)

        relation_names = list(rel_cfg.get("RELATION_NAMES", []))
        self.logic_pairs = self._build_logic_pairs(relation_names)
        glove_init, _ = _build_prototype_text_init(
            relation_names,
            str(rel_cfg.get("SEMANTIC_GLOVE_PATH", "")),
            int(rel_cfg.get("EMBED_DIM", 200)),
            str(rel_cfg.get("RPCM_PROTO_INIT", "semantic")),
            modifier_aware=bool(rel_cfg.get("PROTO_TEXT_INIT_MODIFIER_AWARE", True)),
        )
        proto_args, _ = _build_proto_head_args(cfg, relation_names)
        self.proto_head = PredicatePrototypeHead(
            self.mlp_dim, self.mlp_dim, glove_init,
            ema_alpha=float(rel_cfg.get("RPCM_PROTO_MOMENTUM", 0.9)), **proto_args,
        )

    @staticmethod
    def _build_logic_pairs(relation_names):
        opposites = (("same", "different"), ("directly", "indirectly"), ("positive", "negative"), ("safe", "danger"))
        pairs = set()
        lowered = [str(name).lower().replace("_", " ") for name in relation_names]
        for left_word, right_word in opposites:
            left = {}
            right = {}
            for index, name in enumerate(lowered):
                tokens = name.split()
                if left_word in tokens:
                    left[" ".join(token for token in tokens if token != left_word)] = index
                if right_word in tokens:
                    right[" ".join(token for token in tokens if token != right_word)] = index
            pairs.update((left[key], right[key]) for key in left.keys() & right.keys())
        return tuple(sorted(pairs))

    @staticmethod
    def _box_data(proposal):
        boxes = proposal.bbox.float()
        if proposal.mode == "xywha":
            centers, sizes, angles = boxes[:, :2], boxes[:, 2:4].clamp(min=1e-6), boxes[:, 4]
        else:
            boxes = proposal.convert("xyxy").bbox.float()
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6)
            angles = boxes.new_zeros((len(boxes),))
        scale = boxes.new_tensor([max(float(proposal.size[0]), 1), max(float(proposal.size[1]), 1)])
        return centers / scale, sizes / scale, angles

    def _typed_features(self, proposal, pairs: torch.Tensor):
        centers, sizes, angles = self._box_data(proposal)
        labels = proposal.get_field("labels").long().clamp(0, self.num_obj_classes)
        h, t = pairs[:, 0].long(), pairs[:, 1].long()
        delta = centers[t] - centers[h]
        dist = delta.norm(dim=1).clamp(min=1e-6)
        area_h, area_t = sizes[h].prod(1).clamp(min=1e-6), sizes[t].prod(1).clamp(min=1e-6)
        angle_delta = angles[t] - angles[h]
        geometry = torch.stack((
            delta[:, 0], delta[:, 1], dist,
            delta[:, 0] / sizes[h, 0], delta[:, 1] / sizes[h, 1],
            delta[:, 0] / sizes[t, 0], delta[:, 1] / sizes[t, 1],
            (sizes[t, 0] / sizes[h, 0]).log(), (sizes[t, 1] / sizes[h, 1]).log(),
            (area_t / area_h).log(), angle_delta.sin(), angle_delta.cos(),
            sizes[h].norm(dim=1), sizes[t].norm(dim=1),
        ), dim=1)

        degree_out = torch.bincount(h, minlength=len(proposal)).float()
        degree_in = torch.bincount(t, minlength=len(proposal)).float()
        group = torch.stack((degree_out[h], degree_in[h], degree_out[t], degree_in[t]), dim=1)
        group = torch.log1p(group)
        motion = torch.stack((delta[:, 0], delta[:, 1], dist, angle_delta.sin(), angle_delta.cos(), area_t / area_h), dim=1)
        network = torch.cat((
            self.network_label_embed(labels[h]), self.network_label_embed(labels[t]),
            self._sparse_path_features(pairs, len(proposal), centers.device, centers.dtype),
        ), dim=1)

        same_prob = centers.new_zeros((len(pairs),))
        conf_h = centers.new_zeros((len(pairs),))
        conf_t = centers.new_zeros((len(pairs),))
        anchor_mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.anchor_class_ids:
            anchor_mask |= labels == class_id
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).flatten()
        incidence_anchor = pairs.new_full((len(pairs),), -1)
        if anchor_indices.numel() > 0:
            anchor_ids, weights, assignment_conf = self._assign_anchors(
                centers, sizes, angles, labels, anchor_indices
            )
            ids_h, ids_t = anchor_ids[h], anchor_ids[t]
            w_h, w_t = weights[h], weights[t]
            same_prob = ((ids_h[:, :, None] == ids_t[:, None, :]).float() * w_h[:, :, None] * w_t[:, None, :]).sum((1, 2))
            conf_h, conf_t = assignment_conf[h], assignment_conf[t]
            incidence_anchor = ids_h[:, 0]
        anchor = torch.stack((same_prob, 1 - same_prob, conf_h, conf_t), dim=1)
        experts = torch.stack((
            self.geometry_expert(geometry),
            self.anchor_expert(anchor),
            self.group_expert(group),
            self.motion_expert(motion),
            self.network_expert(network),
        ), dim=1)
        reliable_anchor = (conf_h > 0.35) & (conf_t > 0.35)
        return experts, same_prob, reliable_anchor, incidence_anchor

    def _assign_anchors(self, centers, sizes, angles, labels, anchor_indices):
        """Blockwise learned top-k assignment; temporary memory is O(N * 256)."""
        num_entities = len(centers)
        k = min(self.anchor_topk, len(anchor_indices))
        best_scores = centers.new_full((num_entities, k), -float("inf"))
        best_ids = torch.zeros((num_entities, k), dtype=torch.long, device=centers.device)
        entity_area = sizes.prod(1).clamp(min=1e-6)
        for start in range(0, len(anchor_indices), 256):
            ids = anchor_indices[start:start + 256]
            delta = centers[:, None, :] - centers[ids][None, :, :]
            distance = delta.norm(dim=2, keepdim=True)
            left_top = centers[:, None, :] - sizes[:, None, :] / 2
            left_bottom = centers[:, None, :] + sizes[:, None, :] / 2
            right_top = centers[ids][None, :, :] - sizes[ids][None, :, :] / 2
            right_bottom = centers[ids][None, :, :] + sizes[ids][None, :, :] / 2
            intersection = (torch.minimum(left_bottom, right_bottom) - torch.maximum(left_top, right_top)).clamp(min=0)
            overlap = intersection.prod(2, keepdim=True) / entity_area[:, None, None]
            angle_similarity = (angles[:, None] - angles[ids][None, :]).cos().unsqueeze(2)
            area_ratio = (entity_area[:, None] / entity_area[ids][None, :]).log().unsqueeze(2)
            anchor_embed = self.anchor_label_embed(labels[ids]).unsqueeze(0).expand(num_entities, -1, -1)
            entity_embed = self.anchor_label_embed(labels).unsqueeze(1).expand(-1, len(ids), -1)
            assignment_features = torch.cat((
                overlap, distance, angle_similarity, area_ratio, entity_embed, anchor_embed,
            ), dim=2)
            scores = self.anchor_assignment(assignment_features).squeeze(2)
            candidate_scores = torch.cat((best_scores, scores), dim=1)
            candidate_ids = torch.cat((best_ids, ids[None, :].expand(num_entities, -1)), dim=1)
            best_scores, selected = candidate_scores.topk(k, dim=1)
            best_ids = candidate_ids.gather(1, selected)
        confidence = torch.sigmoid(best_scores[:, 0])
        weights = torch.softmax(best_scores, dim=1) * confidence[:, None]
        return best_ids, weights, confidence

    @staticmethod
    def _sparse_path_features(pairs, num_nodes: int, device, dtype):
        """Direct/reverse and bounded 2/3-hop reachability without an N x N map."""
        adjacency = [set() for _ in range(num_nodes)]
        pair_list = [(int(h), int(t)) for h, t in pairs.detach().cpu().tolist()]
        for head, tail in pair_list:
            adjacency[head].add(tail)
        features = []
        for head, tail in pair_list:
            reverse = float(head in adjacency[tail])
            two_hop = any(tail in adjacency[mid] for mid in adjacency[head])
            three_hop = False
            if not two_hop:
                three_hop = any(
                    tail in adjacency[mid2]
                    for mid1 in adjacency[head]
                    for mid2 in adjacency[mid1]
                )
            shared_next = len(adjacency[head].intersection(adjacency[tail]))
            features.append((1.0, reverse, float(two_hop), float(three_hop or shared_next > 0)))
        if not features:
            return torch.zeros((0, 4), device=device, dtype=dtype)
        return torch.tensor(features, device=device, dtype=dtype)

    @staticmethod
    def _incidence(pairs: torch.Tensor, anchors: torch.Tensor):
        rel_ids = torch.arange(len(pairs), device=pairs.device)
        entities = torch.cat((pairs[:, 0], pairs[:, 1]))
        factors = torch.cat((rel_ids, rel_ids))
        valid = anchors >= 0
        if valid.any():
            entities = torch.cat((entities, anchors[valid]))
            factors = torch.cat((factors, rel_ids[valid]))
        return torch.stack((entities, factors), dim=0)

    def _hierarchy_logits(self, hidden: torch.Tensor):
        family_log_prob = F.log_softmax(self.family_head(hidden), dim=1)
        logits = hidden.new_full((len(hidden), self.num_rel_classes), -1e4)
        logits[:, 0] = family_log_prob[:, 0]
        for family, predicates in FAMILY_PREDICATES.items():
            if family == 0:
                continue
            fine = F.log_softmax(self.fine_heads[str(family)](hidden), dim=1)
            logits[:, list(predicates)] = family_log_prob[:, family:family + 1] + fine
        return logits

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        del rel_binarys, logger
        obj_features, rel_features = self.pairwise_feature_extractor(
            roi_features, union_features, proposals, rel_pair_idxs
        )
        num_objs = [len(p) for p in proposals]
        num_rels = [len(p) for p in rel_pair_idxs]
        obj_chunks = obj_features.split(num_objs) if num_objs else []
        rel_chunks = rel_features.split(num_rels) if num_rels else []
        graph_rel_chunks, same_chunks, anchor_valid_chunks = [], [], []
        for proposal, pairs, obj_chunk, rel_chunk in zip(proposals, rel_pair_idxs, obj_chunks, rel_chunks):
            entities = self.obj_to_graph(obj_chunk)
            relations = self.rel_to_graph(rel_chunk)
            experts, same_prob, reliable_anchor, anchors = self._typed_features(proposal, pairs)
            family_gate = F.softmax(self.expert_family_gate(relations), dim=1)
            # geometry, anchor(shared+different), group, motion and network gates.
            expert_gates = torch.stack((
                family_gate[:, 1], family_gate[:, 2] + family_gate[:, 3],
                family_gate[:, 4], family_gate[:, 5], family_gate[:, 6],
            ), dim=1)
            relations = relations + self.expert_norm((experts * expert_gates[:, :, None]).sum(dim=1))
            partition_ids = proposal.get_field("pair_partition_ids") if proposal.has_field("pair_partition_ids") else pairs.new_zeros((len(pairs),))
            output = relations.clone()
            for partition in torch.unique(partition_ids).tolist():
                mask = partition_ids == partition
                part_pairs, part_rel = pairs[mask], relations[mask]
                part_entities = entities
                for layer in self.sparse_layers:
                    part_entities, part_rel = layer(part_entities, part_rel, part_pairs)
                if self.hypergraph_enabled:
                    part_anchors = anchors[mask]
                    incidence = self._incidence(part_pairs, part_anchors)
                    for layer in self.hyper_layers:
                        part_entities, part_rel = layer(part_entities, part_rel, incidence)
                output[mask] = part_rel
            graph_rel_chunks.append(output)
            same_chunks.append(same_prob)
            anchor_valid_chunks.append(reliable_anchor)
        graph_rel = torch.cat(graph_rel_chunks) if graph_rel_chunks else rel_features.new_zeros((0, self.graph_dim))
        base_hidden = self.down_samp(rel_features)
        hidden = self.rel_norm(self.rel_residual(base_hidden) + base_hidden + self.graph_to_hidden(graph_rel))
        hierarchy_logits = self._hierarchy_logits(hidden)
        flat_labels = torch.cat(rel_labels) if self.training and rel_labels else None
        proto_logits, proto_losses = self.proto_head(hidden, flat_labels)
        logits = self.hierarchy_weight * hierarchy_logits + self.proto_logit_weight * proto_logits
        logits = list(logits.split(num_rels))

        add_losses = dict(proto_losses)
        if self.training and flat_labels is not None and flat_labels.numel() > 0:
            family_targets = self.predicate_to_family[flat_labels.clamp(0, self.num_rel_classes - 1)]
            add_losses["loss_family"] = F.cross_entropy(self.family_head(hidden), family_targets) * self.family_loss_weight
            same_prob = torch.cat(same_chunks).clamp(1e-5, 1 - 1e-5)
            reliable_anchor = torch.cat(anchor_valid_chunks)
            anchor_mask = ((family_targets == 2) | (family_targets == 3)) & reliable_anchor
            if anchor_mask.any():
                anchor_targets = (family_targets[anchor_mask] == 2).float()
                add_losses["loss_anchor"] = F.binary_cross_entropy(same_prob[anchor_mask], anchor_targets) * self.anchor_loss_weight
            probabilities = F.softmax(torch.cat(logits), dim=1)
            logic_terms = [probabilities[:, a] * probabilities[:, b] for a, b in self.logic_pairs]
            add_losses["loss_logic"] = (
                torch.stack(logic_terms, dim=1).mean() if logic_terms else probabilities.sum() * 0.0
            ) * self.logic_loss_weight

        refine_logits = [
            proposal.get_field("predict_logits") if proposal.has_field("predict_logits")
            else obj_features.new_zeros((len(proposal), 0)) for proposal in proposals
        ]
        return logits, refine_logits, add_losses
