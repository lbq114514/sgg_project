from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class SemanticPairFilter(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.enabled = bool(rel_cfg.get("SEMA_F_ENABLED", False))
        self.path = Path(str(rel_cfg.get("SEMA_F_PATH", "pretrained/SF_list.json")))
        self.register_buffer("pair_prior", torch.zeros((0, 0, 0), dtype=torch.float32), persistent=False)
        self.loaded = False
        self._load_prior()

    def _load_prior(self) -> None:
        if not self.enabled or not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        self.pair_prior = torch.tensor(prior, dtype=torch.float32)
        self.loaded = True

    def filter_pairs(self, pair_idx: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.loaded or pair_idx.numel() == 0:
            return pair_idx
        prior = self.pair_prior.to(device=pair_idx.device)
        heads = labels[pair_idx[:, 0]].long().clamp(min=0, max=prior.size(0) - 1)
        tails = labels[pair_idx[:, 1]].long().clamp(min=0, max=prior.size(1) - 1)
        allowed = prior[heads, tails].sum(dim=1) > 0
        return pair_idx[allowed]
