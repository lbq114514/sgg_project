from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.structures.boxlist import BoxList
from sgg.utils.geometry import pair_geometry_features


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeMessageBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.msg = MLP(node_dim * 2 + edge_dim, edge_dim, node_dim, 2)
        self.node_upd = MLP(node_dim * 2, node_dim, node_dim, 2)
        self.edge_upd = MLP(node_dim * 2 + edge_dim, edge_dim, edge_dim, 2)

    def forward(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, pair_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = pair_idx[:, 0], pair_idx[:, 1]
        pair_node = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        msg = self.msg(pair_node)
        agg = torch.zeros_like(node_feat)
        agg.index_add_(0, dst, msg)
        deg = torch.bincount(dst, minlength=node_feat.size(0)).float().unsqueeze(-1).clamp(min=1.0).to(node_feat.device)
        agg = agg / deg
        node_feat = self.node_upd(torch.cat([node_feat, agg], dim=-1))
        edge_feat = self.edge_upd(torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1))
        return node_feat, edge_feat


class SubgraphTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, depth: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, batch_first=True)
        self.net = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentSubgraphRelationHead(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.node_dim = cfg["NODE_DIM"]
        self.edge_dim = cfg["EDGE_DIM"]
        self.hidden_dim = cfg["HIDDEN_DIM"]
        self.num_predicates = cfg["NUM_PREDICATES"]
        self.num_subgraphs = cfg["NUM_SUBGRAPHS"]
        self.max_rel_pairs = cfg["MAX_REL_PAIRS"]
        geom_dim_hbb = 6
        geom_dim_obb = 7
        self.node_ctx = MLP(self.node_dim, self.hidden_dim, self.node_dim, 2)
        self.edge_encoder_hbb = MLP(self.node_dim * 2 + geom_dim_hbb, self.hidden_dim, self.edge_dim, 2)
        self.edge_encoder_obb = MLP(self.node_dim * 2 + geom_dim_obb, self.hidden_dim, self.edge_dim, 2)
        self.subgraph_assign = MLP(self.node_dim * 2, self.hidden_dim, self.num_subgraphs, 2)
        self.intra_blocks = nn.ModuleList([EdgeMessageBlock(self.node_dim, self.edge_dim) for _ in range(cfg["NUM_INTRA_LAYERS"])])
        self.inter_subgraph = SubgraphTransformer(self.node_dim, cfg["NUM_ATTENTION_HEADS"], cfg["NUM_INTER_LAYERS"])
        self.node_feedback = MLP(self.node_dim * 2, self.hidden_dim, self.node_dim, 2)
        self.edge_refine = MLP(self.edge_dim + self.node_dim * 2, self.hidden_dim, self.edge_dim, 2)
        self.rel_classifier = nn.Linear(self.edge_dim, self.num_predicates)

    def _build_pair_index(self, n: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(n, device=device)
        grid_i, grid_j = torch.meshgrid(idx, idx, indexing="ij")
        pair_idx = torch.stack([grid_i.reshape(-1), grid_j.reshape(-1)], dim=-1)
        pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
        if pair_idx.size(0) > self.max_rel_pairs:
            keep = torch.randperm(pair_idx.size(0), device=device)[: self.max_rel_pairs]
            pair_idx = pair_idx[keep]
        return pair_idx

    def _encode_edges(self, node_feat: torch.Tensor, boxlist: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        geom = pair_geometry_features(boxlist)
        src, dst = pair_idx[:, 0], pair_idx[:, 1]
        pair = torch.cat([node_feat[src], node_feat[dst], geom[src, dst]], dim=-1)
        if boxlist.mode in {"xyxy", "xywh"}:
            return self.edge_encoder_hbb(pair)
        return self.edge_encoder_obb(pair)

    def _latent_subgraphs(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, pair_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, _ = pair_idx[:, 0], pair_idx[:, 1]
        agg = torch.zeros_like(node_feat)
        agg.index_add_(0, src, edge_feat)
        deg = torch.bincount(src, minlength=node_feat.size(0)).float().unsqueeze(-1).clamp(min=1.0).to(node_feat.device)
        agg = agg / deg
        assign_logits = self.subgraph_assign(torch.cat([node_feat, agg], dim=-1))
        assign = F.softmax(assign_logits, dim=-1)
        z = assign.transpose(0, 1) @ node_feat
        norm = assign.sum(dim=0, keepdim=True).transpose(0, 1).clamp(min=1e-6)
        z = z / norm
        return assign, z

    def _feedback_to_nodes(self, node_feat: torch.Tensor, assign: torch.Tensor, subgraphs: torch.Tensor) -> torch.Tensor:
        node_ctx = assign @ subgraphs
        return self.node_feedback(torch.cat([node_feat, node_ctx], dim=-1))

    def _compute_rel_loss(self, rel_logits: torch.Tensor, pair_idx: torch.Tensor, boxlist: BoxList) -> torch.Tensor:
        if not boxlist.has_field("pair_labels"):
            return rel_logits.sum() * 0.0
        pair_labels = boxlist.get_field("pair_labels")
        labels = pair_labels[pair_idx[:, 0], pair_idx[:, 1]].long().clamp(min=0, max=self.num_predicates - 1)
        return F.cross_entropy(rel_logits, labels)

    def forward(self, node_feats: List[torch.Tensor], proposals: List[BoxList]) -> Dict:
        losses = []
        predictions = []
        aux_losses = []

        for node_feat, boxlist in zip(node_feats, proposals):
            if node_feat.numel() == 0 or len(boxlist) <= 1:
                rel_logits = torch.zeros((0, self.num_predicates), device=node_feat.device)
                pair_idx = torch.zeros((0, 2), dtype=torch.long, device=node_feat.device)
                assign = torch.zeros((len(boxlist), self.num_subgraphs), device=node_feat.device)
                losses.append(rel_logits.sum() * 0.0)
                aux_losses.append(rel_logits.sum() * 0.0)
                predictions.append({"pair_idx": pair_idx, "rel_logits": rel_logits, "subgraph_assign": assign})
                continue

            node_feat = self.node_ctx(node_feat)
            pair_idx = self._build_pair_index(node_feat.size(0), node_feat.device)
            edge_feat = self._encode_edges(node_feat, boxlist, pair_idx)

            for blk in self.intra_blocks:
                node_feat, edge_feat = blk(node_feat, edge_feat, pair_idx)

            assign, subgraphs = self._latent_subgraphs(node_feat, edge_feat, pair_idx)
            subgraphs = self.inter_subgraph(subgraphs.unsqueeze(0)).squeeze(0)
            node_feat = self._feedback_to_nodes(node_feat, assign, subgraphs)

            src, dst = pair_idx[:, 0], pair_idx[:, 1]
            edge_feat = self.edge_refine(torch.cat([edge_feat, node_feat[src], node_feat[dst]], dim=-1))
            rel_logits = self.rel_classifier(edge_feat)

            loss_rel = self._compute_rel_loss(rel_logits, pair_idx, boxlist)
            loss_entropy = -(assign * (assign.clamp(min=1e-8).log())).sum(dim=-1).mean()
            losses.append(loss_rel)
            aux_losses.append(loss_entropy)
            predictions.append({
                "pair_idx": pair_idx,
                "rel_logits": rel_logits,
                "subgraph_assign": assign,
            })

        loss_rel = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=node_feats[0].device)
        loss_aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=node_feats[0].device)
        return {
            "loss_rel": loss_rel,
            "loss_aux": 0.01 * loss_aux,
            "predictions": predictions,
        }
