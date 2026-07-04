from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = os.environ.get(
    "STAR_SGG_ROOT",
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgg.data.sources.star import STARSource
from sgg.modeling.roi_heads.pair_proposal_network import (
    PairProposalNetwork,
    load_glove_class_matrix,
)
from sgg.structures.boxes import BoxList


def parse_args():
    parser = argparse.ArgumentParser(description="Train the detector-independent STAR PairProposalNetwork.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--glove", default="glove/glove.6B.200d.pt")
    parser.add_argument("--output-dir", default="outputs/star_pair_proposal_network")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument(
        "--hard-negative-pool", type=int, default=4,
        help="Candidate multiplier for the hard half of sampled negatives.",
    )
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--val-limit", type=int, default=-1)
    parser.add_argument("--label-dim", type=int, default=32)
    parser.add_argument("--box-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--anchor-classes", default="apron,truck_parking,car_parking,dock,runway,taxiway,breakwater,goods_yard",
    )
    parser.add_argument("--anchor-dim", type=int, default=32)
    parser.add_argument("--predicate-loss-weight", type=float, default=0.2)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress(iterable, **kwargs):
    return tqdm(iterable, disable=not sys.stderr.isatty(), dynamic_ncols=True, **kwargs)


def build_source(data_root: Path) -> STARSource:
    return STARSource.from_paths(
        image_root=data_root / "STAR_img",
        roidb_file=data_root / "STAR-SGG-with-attri.h5",
        dict_file=data_root / "STAR-SGG-dicts-with-attri.json",
        image_file=data_root / "STAR_image_data_v1.json",
        image_ext=".png",
    )


def build_records(source: STARSource, split: str, limit: int):
    records = source.get_split_records(
        split=split,
        box_mode="obb",
        split_mode="fixed",
        filter_empty_relations=True,
        filter_non_overlap=False,
        num_im=limit,
    )
    return records


def record_to_proposal(record: dict, device: torch.device) -> BoxList:
    boxes = torch.as_tensor(record["boxes"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(record["labels"], dtype=torch.long, device=device)
    proposal = BoxList(boxes, (int(record["width"]), int(record["height"])), mode="xywha")
    proposal.add_field("labels", labels)
    return proposal


def unique_positive_pairs(record: dict, device: torch.device) -> torch.Tensor:
    relations = np.asarray(record["relations"])
    if relations.size == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    pairs = sorted({(int(row[0]), int(row[1])) for row in relations if int(row[0]) != int(row[1])})
    return torch.tensor(pairs, dtype=torch.long, device=device)


def predicate_targets_for_pairs(
    record: dict, pair_idx: torch.Tensor, num_predicates: int,
) -> torch.Tensor:
    relation_map: dict[tuple[int, int], set[int]] = {}
    for head, tail, predicate in record["relations"]:
        if int(head) != int(tail) and int(predicate) > 0:
            relation_map.setdefault((int(head), int(tail)), set()).add(int(predicate) - 1)
    targets = torch.zeros(
        (len(pair_idx), num_predicates), dtype=torch.float32, device=pair_idx.device
    )
    for row, pair in enumerate(pair_idx.detach().cpu().tolist()):
        for predicate in relation_map.get((int(pair[0]), int(pair[1])), ()):
            if 0 <= predicate < num_predicates:
                targets[row, predicate] = 1.0
    return targets


def load_semantic_pair_support(path: str | Path, num_classes: int) -> torch.Tensor:
    """Return class pairs known to support at least one predicate."""
    with Path(path).open("r", encoding="utf-8") as handle:
        prior = torch.as_tensor(json.load(handle), dtype=torch.float32)
    if prior.ndim != 3 or prior.size(0) != num_classes or prior.size(1) != num_classes:
        raise ValueError(
            f"Semantic support must have shape ({num_classes}, {num_classes}, predicates), "
            f"got {tuple(prior.shape)}"
        )
    return prior.gt(0).any(dim=2)


def _sample_unique_negative_pairs(
    num_entities: int,
    count: int,
    positive_pairs: Iterable[tuple[int, int]],
    rng: random.Random,
    labels: Sequence[int] | None = None,
    forbidden_label_pairs: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    """Sample negatives without materializing an N x N mask or unbounded retries."""
    if num_entities <= 1 or count <= 0:
        return []
    positives = set(positive_pairs)
    if forbidden_label_pairs is not None:
        if labels is None or len(labels) != num_entities:
            raise ValueError("Entity labels are required with forbidden_label_pairs")
        forbidden_label_pairs = forbidden_label_pairs.cpu().bool()
        labels = [int(label) for label in labels]
    if forbidden_label_pairs is None:
        max_negatives = num_entities * (num_entities - 1) - len(positives)
    else:
        label_counts: dict[int, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        max_negatives = 0
        for head_label, head_count in label_counts.items():
            for tail_label, tail_count in label_counts.items():
                if bool(forbidden_label_pairs[head_label, tail_label]):
                    continue
                max_negatives += head_count * tail_count
                if head_label == tail_label:
                    max_negatives -= head_count
        max_negatives -= sum(
            1 for head, tail in positives
            if not bool(forbidden_label_pairs[labels[head], labels[tail]])
        )
    target = min(int(count), max(max_negatives, 0))
    if target <= 0:
        return []

    # When nearly every eligible pair is requested, direct enumeration is both
    # bounded and much faster than waiting for the last unseen random samples.
    if forbidden_label_pairs is not None and target * 2 >= max_negatives:
        entities_by_label: dict[int, list[int]] = {}
        for entity_id, label in enumerate(labels):
            entities_by_label.setdefault(label, []).append(entity_id)
        eligible = []
        for head_label, heads in entities_by_label.items():
            for tail_label, tails in entities_by_label.items():
                if bool(forbidden_label_pairs[head_label, tail_label]):
                    continue
                eligible.extend(
                    (head, tail) for head in heads for tail in tails
                    if head != tail and (head, tail) not in positives
                )
        if target >= len(eligible):
            rng.shuffle(eligible)
            return eligible
        return rng.sample(eligible, target)

    selected: set[tuple[int, int]] = set()
    max_attempts = max(target * 50, 1000)
    attempts = 0
    while len(selected) < target and attempts < max_attempts:
        remaining = target - len(selected)
        for _ in range(max(remaining * 2, 32)):
            attempts += 1
            head = rng.randrange(num_entities)
            tail = rng.randrange(num_entities - 1)
            if tail >= head:
                tail += 1
            pair = (head, tail)
            label_forbidden = (
                forbidden_label_pairs is not None
                and bool(forbidden_label_pairs[labels[head], labels[tail]])
            )
            if pair not in positives and not label_forbidden:
                selected.add(pair)
            if len(selected) >= target:
                break
            if attempts >= max_attempts:
                break
    return list(selected)


@torch.no_grad()
def sample_negative_pairs(
    model: PairProposalNetwork,
    proposal: BoxList,
    positive_pairs: torch.Tensor,
    negative_ratio: int,
    hard_pool_multiplier: int,
    rng: random.Random,
    score_chunk_size: int,
    forbidden_label_pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    num_positive = int(positive_pairs.size(0))
    requested = max(num_positive * int(negative_ratio), 1)
    hard_count = requested // 2
    random_count = requested - hard_count
    positives = {(int(h), int(t)) for h, t in positive_pairs.cpu().tolist()}
    pool_count = hard_count * max(int(hard_pool_multiplier), 1)
    sampled = _sample_unique_negative_pairs(
        len(proposal), pool_count + random_count, positives, rng,
        labels=proposal.get_field("labels").cpu().tolist(),
        forbidden_label_pairs=forbidden_label_pairs,
    )
    if not sampled:
        return torch.zeros((0, 2), dtype=torch.long, device=proposal.bbox.device)
    pool = sampled[:pool_count]
    random_negatives = sampled[pool_count: pool_count + random_count]
    if hard_count > 0 and pool:
        pool_tensor = torch.tensor(pool, dtype=torch.long, device=proposal.bbox.device)
        score_parts = []
        for start in range(0, len(pool_tensor), score_chunk_size):
            score_parts.append(model.score_pairs(proposal, pool_tensor[start:start + score_chunk_size]))
        pool_scores = torch.cat(score_parts)
        hard_idx = pool_scores.topk(min(hard_count, len(pool_tensor))).indices.cpu().tolist()
        hard_negatives = [pool[index] for index in hard_idx]
    else:
        hard_negatives = []
    negatives = hard_negatives + random_negatives
    return torch.tensor(negatives, dtype=torch.long, device=proposal.bbox.device)


def train_record(
    model: PairProposalNetwork,
    optimizer: torch.optim.Optimizer,
    record: dict,
    device: torch.device,
    args,
    rng: random.Random,
    forbidden_label_pairs: torch.Tensor | None = None,
) -> dict[str, float]:
    proposal = record_to_proposal(record, device)
    positive_pairs = unique_positive_pairs(record, device)
    if positive_pairs.numel() == 0 or len(proposal) <= 1:
        return {"loss": 0.0, "positive_loss": 0.0, "negative_loss": 0.0, "pairs": 0.0}
    model.eval()
    negative_pairs = sample_negative_pairs(
        model, proposal, positive_pairs, args.negative_ratio,
        args.hard_negative_pool, rng, args.block_size, forbidden_label_pairs,
    )
    pair_idx = torch.cat((positive_pairs, negative_pairs), dim=0)
    targets = torch.cat((
        torch.ones(len(positive_pairs), device=device),
        torch.zeros(len(negative_pairs), device=device),
    ))
    predicate_targets = predicate_targets_for_pairs(
        record, pair_idx, model.num_predicates
    ) if model.num_predicates > 0 else None
    order = torch.randperm(len(pair_idx), device=device)
    pair_idx, targets = pair_idx[order], targets[order]
    if predicate_targets is not None:
        predicate_targets = predicate_targets[order]
    positive_weight = float(len(negative_pairs)) / max(float(len(positive_pairs)), 1.0)
    pos_weight = torch.tensor(positive_weight, device=device)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    total_loss_sum = 0.0
    positive_loss_sum = 0.0
    negative_loss_sum = 0.0
    predicate_loss_sum = 0.0
    ranking_loss_sum = 0.0
    total_pairs = max(len(pair_idx), 1)
    for start in range(0, total_pairs, args.block_size):
        end = min(start + args.block_size, total_pairs)
        chunk_targets = targets[start:end]
        logits, predicate_logits = model.pair_outputs(proposal, pair_idx[start:end])
        raw = F.binary_cross_entropy_with_logits(
            logits, chunk_targets, pos_weight=pos_weight, reduction="none"
        )
        chunk_loss = raw.sum() / total_pairs
        if predicate_logits is not None:
            aux = F.binary_cross_entropy_with_logits(
                predicate_logits, predicate_targets[start:end], reduction="mean"
            )
            predicate_loss_sum += float(aux.detach()) * ((end - start) / total_pairs)
            chunk_loss = chunk_loss + getattr(args, "predicate_loss_weight", 0.2) * aux * ((end - start) / total_pairs)
        positive_logits = logits[chunk_targets > 0]
        negative_logits = logits[chunk_targets == 0]
        if positive_logits.numel() > 0 and negative_logits.numel() > 0:
            ranking = F.relu(
                getattr(args, "ranking_margin", 0.2) - positive_logits.mean() + negative_logits.mean()
            )
            ranking_loss_sum += float(ranking.detach()) * ((end - start) / total_pairs)
            chunk_loss = chunk_loss + getattr(args, "ranking_loss_weight", 0.2) * ranking * ((end - start) / total_pairs)
        chunk_loss.backward()
        total_loss_sum += float(raw.detach().sum())
        positive_loss_sum += float(raw[chunk_targets > 0].detach().sum())
        negative_loss_sum += float(raw[chunk_targets == 0].detach().sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return {
        "loss": total_loss_sum / total_pairs,
        "positive_loss": positive_loss_sum / max(len(positive_pairs), 1),
        "negative_loss": negative_loss_sum / max(len(negative_pairs), 1),
        "predicate_loss": predicate_loss_sum,
        "ranking_loss": ranking_loss_sum,
        "pairs": float(total_pairs),
    }


@torch.no_grad()
def evaluate(
    model: PairProposalNetwork,
    records: Sequence[dict],
    device: torch.device,
    topk: int,
    block_size: int,
) -> dict[str, float]:
    model.eval()
    total_gt = 0
    total_hits = 0
    total_candidates = 0
    evaluated = 0
    started = time.perf_counter()
    progress = _progress(records, desc="validation", leave=False)
    for record in progress:
        proposal = record_to_proposal(record, device)
        positives = unique_positive_pairs(record, device)
        if positives.numel() == 0:
            continue
        predicted, _ = model.topk_pairs(proposal, topk=topk, block_size=block_size)
        predicted_set = {(int(h), int(t)) for h, t in predicted.cpu().tolist()}
        gt_pairs = {(int(h), int(t)) for h, t in positives.cpu().tolist()}
        total_hits += len(gt_pairs & predicted_set)
        total_gt += len(gt_pairs)
        total_candidates += len(predicted)
        evaluated += 1
        progress.set_postfix(recall=f"{total_hits / max(total_gt, 1):.4f}")
    return {
        "pair_recall": total_hits / max(total_gt, 1),
        "gt_pairs": float(total_gt),
        "gt_hits": float(total_hits),
        "avg_candidates": total_candidates / max(evaluated, 1),
        "images": float(evaluated),
        "seconds": time.perf_counter() - started,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, metadata: dict) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        **metadata,
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(data_root)
    train_records = build_records(source, "train", args.train_limit)
    val_records = build_records(source, "val", args.val_limit)
    class_names = list(source.ind_to_classes)
    anchor_names = [name.strip() for name in args.anchor_classes.split(",") if name.strip()]
    class_to_id = {name: index for index, name in enumerate(class_names)}
    missing_anchors = [name for name in anchor_names if name not in class_to_id]
    if missing_anchors:
        raise ValueError(f"Unknown anchor classes: {missing_anchors}")
    anchor_class_ids = [class_to_id[name] for name in anchor_names]
    glove_matrix, glove_diagnostics = load_glove_class_matrix(args.glove, class_names)
    model_config = {
        "num_obj_classes": len(class_names), "label_dim": args.label_dim,
        "box_dim": args.box_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "glove_dim": int(glove_matrix.size(1)),
        "anchor_class_ids": anchor_class_ids, "anchor_class_names": anchor_names,
        "anchor_dim": args.anchor_dim,
        "num_predicates": len(source.ind_to_predicates) - 1,
    }
    model = PairProposalNetwork(
        num_obj_classes=len(class_names), label_dim=args.label_dim,
        box_dim=args.box_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        glove_init=glove_matrix,
        anchor_class_ids=anchor_class_ids, anchor_dim=args.anchor_dim,
        num_predicates=len(source.ind_to_predicates) - 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    metadata = {
        "model_config": model_config,
        "class_names": class_names,
        "glove": {"path": str(args.glove), **glove_diagnostics},
        "training_args": vars(args),
    }
    print(json.dumps({
        "device": str(device), "parameters": model.num_parameters,
        "train_images": len(train_records), "val_images": len(val_records),
        "glove": glove_diagnostics,
    }, ensure_ascii=False), flush=True)

    best_recall = float("-inf")
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        order = list(range(len(train_records)))
        rng.shuffle(order)
        totals = {
            "loss": 0.0, "positive_loss": 0.0, "negative_loss": 0.0,
            "predicate_loss": 0.0, "ranking_loss": 0.0, "pairs": 0.0,
        }
        trained = 0
        progress = _progress(order, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        for index in progress:
            stats = train_record(
                model, optimizer, train_records[index], device, args, rng,
            )
            if stats["pairs"] <= 0:
                continue
            for key in totals:
                totals[key] += stats[key]
            trained += 1
            progress.set_postfix(loss=f"{totals['loss'] / max(trained, 1):.4f}")
        train_metrics = {
            key: value / max(trained, 1) for key, value in totals.items() if key != "pairs"
        }
        train_metrics["sampled_pairs"] = totals["pairs"]
        train_metrics["images"] = float(trained)
        train_metrics["seconds"] = time.perf_counter() - started
        val_metrics = evaluate(model, val_records, device, args.topk, args.block_size)
        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(output_dir / "model_last.pth", model, optimizer, epoch, metrics, metadata)
        if val_metrics["pair_recall"] > best_recall:
            best_recall = val_metrics["pair_recall"]
            save_checkpoint(output_dir / "model_best.pth", model, optimizer, epoch, metrics, metadata)
        print(json.dumps({"epoch": epoch, **metrics, "best_pair_recall": best_recall}), flush=True)


if __name__ == "__main__":
    main()
