from __future__ import annotations

from typing import Sequence

import torch

from sgg.structures.boxes import BoxList
from sgg.structures.boxlist_ops import boxlist_iou


class RelationSamplingProcessor:
    def __init__(self, cfg: dict):
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        roi_heads_cfg = cfg["MODEL"].get("ROI_HEADS", {})
        self.use_gt_box = bool(rel_cfg.get("USE_GT_BOX", False))
        self.batch_size_per_image = int(rel_cfg.get("BATCH_SIZE_PER_IMAGE", 1024))
        self.test_pair_sampler = str(rel_cfg.get("TEST_PAIR_SAMPLER", "ALL")).upper()
        self.test_subgraph_topk = int(rel_cfg.get("TEST_SUBGRAPH_TOPK", 32))
        self.test_subgraph_global_topk = int(rel_cfg.get("TEST_SUBGRAPH_GLOBAL_TOPK", 0))
        self.positive_fraction = float(rel_cfg.get("POSITIVE_FRACTION", 0.25))
        self.require_overlap = bool(rel_cfg.get("REQUIRE_BOX_OVERLAP", False))
        self.fg_thres = float(roi_heads_cfg.get("FG_IOU_THRESHOLD", 0.5))
        self.num_sample_per_gt_rel = int(rel_cfg.get("NUM_SAMPLE_PER_GT_REL", 4))
        self.mbs = int(cfg.get("mbs", max(1, int(self.batch_size_per_image * self.positive_fraction))))

    def prepare_test_pairs(self, device, proposals: Sequence[BoxList]):
        if self.test_pair_sampler in {"SUBGRAPH", "GEOMETRIC_SUBGRAPH"}:
            return self._prepare_geometric_subgraph_test_pairs(device, proposals)
        return self._prepare_all_test_pairs(device, proposals)

    def _prepare_all_test_pairs(self, device, proposals: Sequence[BoxList]):
        rel_pair_idxs = []
        for proposal in proposals:
            n = len(proposal)
            if n <= 1:
                rel_pair_idxs.append(torch.zeros((0, 2), dtype=torch.long, device=device))
                continue
            cand_matrix = torch.ones((n, n), device=device, dtype=torch.bool)
            cand_matrix.fill_diagonal_(False)
            if (not self.use_gt_box) and self.require_overlap:
                overlaps = boxlist_iou(proposal, proposal, mode="obb" if proposal.mode == "xywha" else "hbb")
                cand_matrix = cand_matrix & overlaps.gt(0)
            rel_pair_idxs.append(cand_matrix.nonzero(as_tuple=False))
        return rel_pair_idxs

    def _prepare_geometric_subgraph_test_pairs(self, device, proposals: Sequence[BoxList]):
        rel_pair_idxs = []
        for proposal in proposals:
            n = len(proposal)
            if n <= 1:
                rel_pair_idxs.append(torch.zeros((0, 2), dtype=torch.long, device=device))
                continue

            local_topk = min(max(self.test_subgraph_topk, 0), n - 1)
            global_topk = min(max(self.test_subgraph_global_topk, 0), max(n - 1 - local_topk, 0))
            if local_topk + global_topk == 0:
                rel_pair_idxs.append(torch.zeros((0, 2), dtype=torch.long, device=device))
                continue

            centers, sizes = self._box_centers_and_sizes(proposal, device)
            center_delta = centers[:, None, :] - centers[None, :, :]
            center_dist = center_delta.square().sum(dim=-1).sqrt()
            scale = sizes.prod(dim=1).clamp(min=1e-6).sqrt()
            pair_scale = 0.5 * (scale[:, None] + scale[None, :]).clamp(min=1e-6)
            distance = center_dist / pair_scale
            distance.fill_diagonal_(float("inf"))

            sorted_idx = distance.argsort(dim=1)
            neighbor_chunks = []
            if local_topk > 0:
                neighbor_chunks.append(sorted_idx[:, :local_topk])
            if global_topk > 0:
                remaining = sorted_idx[:, local_topk:]
                if remaining.size(1) <= global_topk:
                    neighbor_chunks.append(remaining)
                else:
                    steps = torch.linspace(
                        0,
                        remaining.size(1) - 1,
                        global_topk,
                        device=device,
                    ).round().long()
                    neighbor_chunks.append(remaining[:, steps])
            neighbor_idx = torch.cat(neighbor_chunks, dim=1)
            topk = neighbor_idx.size(1)
            subj = torch.arange(n, device=device).unsqueeze(1).expand(-1, topk)
            rel_pair_idxs.append(torch.stack([subj.reshape(-1), neighbor_idx.reshape(-1)], dim=1))
        return rel_pair_idxs

    def _box_centers_and_sizes(self, proposal: BoxList, device: torch.device):
        boxes = proposal.bbox.to(device=device, dtype=torch.float32)
        if proposal.mode == "xywha":
            return boxes[:, :2], boxes[:, 2:4].clamp(min=1e-6)
        xyxy = proposal.convert("xyxy").bbox.to(device=device, dtype=torch.float32)
        centers = 0.5 * (xyxy[:, :2] + xyxy[:, 2:])
        sizes = (xyxy[:, 2:] - xyxy[:, :2]).clamp(min=1e-6)
        return centers, sizes

    def gtbox_relsample(self, proposals: Sequence[BoxList], targets: Sequence[BoxList]):
        rel_labels = []
        rel_pair_idxs = []
        rel_binarys = []
        out_props = []
        for proposal, target in zip(proposals, targets):
            pair_idx, labels, binary_rel = self._sample_gtbox_rpcm(proposal, target)
            rel_pair_idxs.append(pair_idx)
            rel_labels.append(labels)
            rel_binarys.append(binary_rel)
            out_props.append(proposal)
        return out_props, rel_labels, rel_pair_idxs, rel_binarys

    def detect_relsample(self, proposals: Sequence[BoxList], targets: Sequence[BoxList]):
        rel_labels = []
        rel_pair_idxs = []
        rel_binarys = []
        out_props = []
        for proposal, target in zip(proposals, targets):
            pair_idx, labels, binary_rel = self._sample_detect_rpcm(proposal, target)
            rel_pair_idxs.append(pair_idx)
            rel_labels.append(labels)
            rel_binarys.append(binary_rel)
            out_props.append(proposal)
        return out_props, rel_labels, rel_pair_idxs, rel_binarys

    def _sample_gtbox_rpcm(self, proposal: BoxList, target: BoxList):
        device = proposal.bbox.device
        num_prp = len(proposal)
        if num_prp <= 1:
            return self._empty(device, num_prp)

        pair_labels = self._get_relation_matrix(target).to(device)
        pos_pair_idx = torch.nonzero(pair_labels > 0, as_tuple=False)
        pos_labels = pair_labels[pos_pair_idx[:, 0], pos_pair_idx[:, 1]].long() if pos_pair_idx.numel() > 0 else torch.zeros((0,), dtype=torch.long, device=device)

        binary_rel = torch.zeros((num_prp, num_prp), device=device, dtype=torch.long)
        if pos_pair_idx.numel() > 0:
            binary_rel[pos_pair_idx[:, 0], pos_pair_idx[:, 1]] = 1
            binary_rel[pos_pair_idx[:, 1], pos_pair_idx[:, 0]] = 1

        rel_possibility = torch.ones((num_prp, num_prp), device=device, dtype=torch.bool)
        rel_possibility.fill_diagonal_(False)
        if pos_pair_idx.numel() > 0:
            rel_possibility[pos_pair_idx[:, 0], pos_pair_idx[:, 1]] = False

        num_pos_per_img = self.mbs
        if pos_pair_idx.size(0) > num_pos_per_img:
            perm = torch.randperm(pos_pair_idx.size(0), device=device)[:num_pos_per_img]
            pos_pair_idx = pos_pair_idx[perm]
            pos_labels = pos_labels[perm]

        bg_pair_idx = torch.nonzero(rel_possibility, as_tuple=False)
        num_fg = min(pos_pair_idx.size(0), num_pos_per_img)
        num_bg = min(max(self.batch_size_per_image - num_fg, 0), bg_pair_idx.size(0))
        if num_bg > 0:
            perm = torch.randperm(bg_pair_idx.size(0), device=device)[:num_bg]
            bg_pair_idx = bg_pair_idx[perm]
        else:
            bg_pair_idx = torch.zeros((0, 2), dtype=torch.long, device=device)
        bg_labels = torch.zeros((bg_pair_idx.size(0),), dtype=torch.long, device=device)

        pair_idx = torch.cat((pos_pair_idx, bg_pair_idx), dim=0)
        labels = torch.cat((pos_labels, bg_labels), dim=0)
        return pair_idx, labels, binary_rel

    def _sample_detect_rpcm(self, proposal: BoxList, target: BoxList):
        device = proposal.bbox.device
        num_prp = len(proposal)
        if num_prp <= 1 or len(target) == 0 or not target.has_field("relation_triplets"):
            return self._empty(device, num_prp)

        prp_labels = proposal.get_field("labels").long().to(device)
        tgt_labels = target.get_field("labels").long().to(device)
        tgt_rel_matrix = self._get_relation_matrix(target).to(device)
        ious = boxlist_iou(target, proposal, mode="obb" if target.mode == "xywha" and proposal.mode == "xywha" else "hbb")
        is_match = (tgt_labels[:, None] == prp_labels[None, :]) & (ious > self.fg_thres)

        if self.require_overlap and (not self.use_gt_box):
            prp_self_iou = boxlist_iou(proposal, proposal, mode="obb" if proposal.mode == "xywha" else "hbb")
            rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)
        else:
            rel_possibility = torch.ones((num_prp, num_prp), device=device, dtype=torch.bool)
            rel_possibility.fill_diagonal_(False)

        fg_mask = prp_labels > 0
        rel_possibility[~fg_mask] = False
        rel_possibility[:, ~fg_mask] = False

        return self._motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match, rel_possibility)

    def _motif_rel_fg_bg_sampling(
        self,
        device: torch.device,
        tgt_rel_matrix: torch.Tensor,
        ious: torch.Tensor,
        is_match: torch.Tensor,
        rel_possibility: torch.Tensor,
    ):
        tgt_pair_idx = torch.nonzero(tgt_rel_matrix > 0, as_tuple=False)
        if tgt_pair_idx.numel() == 0:
            num_prp = is_match.size(1)
            bg_pair_idx = torch.nonzero(rel_possibility, as_tuple=False)
            num_bg = min(self.batch_size_per_image, bg_pair_idx.size(0))
            if num_bg > 0:
                perm = torch.randperm(bg_pair_idx.size(0), device=device)[:num_bg]
                bg_pair_idx = bg_pair_idx[perm]
                return bg_pair_idx, torch.zeros((bg_pair_idx.size(0),), dtype=torch.long, device=device), torch.zeros((num_prp, num_prp), dtype=torch.long, device=device)
            return self._empty(device, num_prp)

        tgt_head_idx = tgt_pair_idx[:, 0].long()
        tgt_tail_idx = tgt_pair_idx[:, 1].long()
        tgt_rel_labels = tgt_rel_matrix[tgt_head_idx, tgt_tail_idx].long()

        num_prp = is_match.size(1)
        binary_head = is_match[tgt_head_idx]
        binary_tail = is_match[tgt_tail_idx]
        binary_rel = torch.zeros((num_prp, num_prp), device=device, dtype=torch.long)
        fg_triplets = []

        for rel_id in range(tgt_rel_labels.size(0)):
            match_head = torch.nonzero(binary_head[rel_id], as_tuple=False).squeeze(1)
            match_tail = torch.nonzero(binary_tail[rel_id], as_tuple=False).squeeze(1)
            if match_head.numel() > 0 and match_tail.numel() > 0:
                head_grid = match_head.view(1, -1).expand(match_tail.numel(), -1).reshape(-1)
                tail_grid = match_tail.view(-1, 1).expand(-1, match_head.numel()).reshape(-1)
                binary_rel[head_grid, tail_grid] = 1
                binary_rel[tail_grid, head_grid] = 1

            prp_head_idx = torch.nonzero(is_match[tgt_head_idx[rel_id]], as_tuple=False).squeeze(1)
            prp_tail_idx = torch.nonzero(is_match[tgt_tail_idx[rel_id]], as_tuple=False).squeeze(1)
            if prp_head_idx.numel() == 0 or prp_tail_idx.numel() == 0:
                continue

            head_grid = prp_head_idx.view(-1, 1).expand(-1, prp_tail_idx.numel()).reshape(-1)
            tail_grid = prp_tail_idx.view(1, -1).expand(prp_head_idx.numel(), -1).reshape(-1)
            valid_pair = head_grid != tail_grid
            if valid_pair.sum().item() <= 0:
                continue

            head_grid = head_grid[valid_pair]
            tail_grid = tail_grid[valid_pair]
            rel_possibility[head_grid, tail_grid] = False

            rel_label = int(tgt_rel_labels[rel_id].item())
            fg_rel_i = torch.stack(
                (
                    head_grid,
                    tail_grid,
                    torch.full((tail_grid.size(0),), rel_label, dtype=torch.long, device=device),
                ),
                dim=1,
            )

            if fg_rel_i.size(0) > self.num_sample_per_gt_rel:
                iou_score = (ious[tgt_head_idx[rel_id], head_grid] * ious[tgt_tail_idx[rel_id], tail_grid]).clamp(min=0)
                if float(iou_score.sum().item()) > 0:
                    perm = torch.multinomial(iou_score / iou_score.sum(), self.num_sample_per_gt_rel, replacement=False)
                else:
                    perm = torch.randperm(fg_rel_i.size(0), device=device)[: self.num_sample_per_gt_rel]
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.numel() > 0:
                fg_triplets.append(fg_rel_i)

        if fg_triplets:
            fg_triplets = torch.cat(fg_triplets, dim=0).long()
            if fg_triplets.size(0) > self.mbs:
                perm = torch.randperm(fg_triplets.size(0), device=device)[: self.mbs]
                fg_triplets = fg_triplets[perm]
        else:
            fg_triplets = torch.zeros((0, 3), dtype=torch.long, device=device)

        bg_pair_idx = torch.nonzero(rel_possibility, as_tuple=False)
        num_neg = min(max(self.batch_size_per_image - fg_triplets.size(0), 0), bg_pair_idx.size(0))
        if num_neg > 0:
            perm = torch.randperm(bg_pair_idx.size(0), device=device)[:num_neg]
            bg_pair_idx = bg_pair_idx[perm]
            bg_triplets = torch.cat(
                (
                    bg_pair_idx,
                    torch.zeros((bg_pair_idx.size(0), 1), dtype=torch.long, device=device),
                ),
                dim=1,
            )
        else:
            bg_triplets = torch.zeros((0, 3), dtype=torch.long, device=device)

        triplets = torch.cat((fg_triplets, bg_triplets), dim=0)
        if triplets.size(0) == 0:
            triplets = torch.zeros((1, 3), dtype=torch.long, device=device)
        return triplets[:, :2], triplets[:, 2].long(), binary_rel

    def _get_relation_matrix(self, boxlist: BoxList) -> torch.Tensor:
        if boxlist.has_field("pair_labels"):
            return boxlist.get_field("pair_labels").long()
        n = len(boxlist)
        matrix = torch.zeros((n, n), dtype=torch.long, device=boxlist.bbox.device)
        if boxlist.has_field("relation_triplets"):
            rel_triplets = boxlist.get_field("relation_triplets").long()
            if rel_triplets.numel() > 0:
                matrix[rel_triplets[:, 0], rel_triplets[:, 1]] = rel_triplets[:, 2]
        return matrix

    def _empty(self, device, n: int):
        return (
            torch.zeros((0, 2), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((n, n), dtype=torch.long, device=device),
        )


def make_roi_relation_samp_processor(cfg: dict):
    return RelationSamplingProcessor(cfg)
