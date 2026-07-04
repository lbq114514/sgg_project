from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = os.environ.get(
    "STAR_SGG_ROOT",
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgg.modeling.roi_heads.pair_proposal_network import load_pair_proposal_checkpoint
from sgg.modeling.roi_heads.ppg import PairProposalGenerator
from sgg.config.defaults import get_default_cfg
from tools.train_pair_proposal_network import (
    build_records,
    build_source,
    load_semantic_pair_support,
    record_to_proposal,
    unique_positive_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PairProposalNetwork.")
    parser.add_argument("--method", choices=("PPN", "PPG", "NONE"), default="PPN")
    parser.add_argument("--run-name", default="", help="Display name used in comparison output.")
    parser.add_argument("--checkpoint", default="outputs/star_pair_proposal_network_smoke/model_best.pth")
    parser.add_argument("--ppg-checkpoint", default="pretrained/STAR_OBB.pth")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--topk", type=int, nargs="+", default=[1000, 2000, 5000, 10000])
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--semantic-support", default="pretrained/SF_list_support.json")
    parser.add_argument("--no-semantic-filter", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


@torch.no_grad()
def _semantic_candidate_pairs(proposal, semantic_support):
    n = len(proposal)
    device = proposal.bbox.device
    if n <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    mask = torch.ones((n, n), dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    pairs = mask.nonzero(as_tuple=False)
    if semantic_support is not None:
        labels = proposal.get_field("labels").long()
        allowed = semantic_support.to(device)
        pairs = pairs[allowed[labels[pairs[:, 0]], labels[pairs[:, 1]]]]
    return pairs


def evaluate_topk(
    model, records, device, topks, block_size, semantic_support=None,
    predicate_names=None, method="PPN",
):
    if model is not None:
        model.eval()
    topks = sorted({int(k) for k in topks if int(k) > 0})
    if not topks:
        raise ValueError("At least one positive top-k value is required")
    hits = {k: 0 for k in topks}
    candidate_totals = {k: 0 for k in topks}
    predicate_totals = defaultdict(int)
    predicate_hits = {k: defaultdict(int) for k in topks}
    total_gt = 0
    base_pair_total = 0
    semantic_pair_total = 0
    evaluated = 0
    started = time.perf_counter()
    progress = tqdm(
        records, desc="test pair recall", leave=True,
        disable=not sys.stderr.isatty(), dynamic_ncols=True,
    )
    for record in progress:
        proposal = record_to_proposal(record, device)
        positives = unique_positive_pairs(record, device)
        if positives.numel() == 0:
            continue
        allowed = semantic_support.to(device) if semantic_support is not None else None
        if method == "PPN":
            predicted, _ = model.topk_pairs(
                proposal, topk=max(topks), block_size=block_size,
                allowed_label_pairs=allowed,
            )
            predicted_by_k = {k: predicted[:k] for k in topks}
        else:
            semantic_pairs = _semantic_candidate_pairs(proposal, semantic_support)
            if method == "PPG":
                original_topk, original_threshold = model.topk, model.threshold
                model.threshold = 0
                model.topk = max(topks)
                predicted = model.filter_pairs(proposal, semantic_pairs)
                predicted_by_k = {k: predicted[:k] for k in topks}
                model.topk, model.threshold = original_topk, original_threshold
            else:
                # NONE means no pair-proposal truncation after semantic filtering.
                predicted_by_k = {k: semantic_pairs for k in topks}
        gt_pairs = {(int(h), int(t)) for h, t in positives.cpu().tolist()}
        labels = proposal.get_field("labels").long()
        num_entities = len(proposal)
        base_pair_total += num_entities * max(num_entities - 1, 0)
        if allowed is None:
            semantic_pair_total += num_entities * max(num_entities - 1, 0)
        else:
            counts = torch.bincount(labels, minlength=allowed.size(0)).long()
            pair_counts = counts[:, None] * counts[None, :]
            pair_counts.diagonal().sub_(counts)
            semantic_pair_total += int(pair_counts[allowed].sum())
        relations = [
            (int(row[0]), int(row[1]), int(row[2]))
            for row in record["relations"]
            if int(row[0]) != int(row[1])
        ]
        for _, _, predicate in relations:
            predicate_totals[predicate] += 1
        for k in topks:
            predicted_k = predicted_by_k[k]
            selected = {(int(h), int(t)) for h, t in predicted_k.cpu().tolist()}
            hits[k] += len(gt_pairs & selected)
            candidate_totals[k] += len(predicted_k)
            for head, tail, predicate in relations:
                if (head, tail) in selected:
                    predicate_hits[k][predicate] += 1
        total_gt += len(gt_pairs)
        evaluated += 1
        progress.set_postfix(
            recall=f"{hits[max(topks)] / max(total_gt, 1):.4f}",
            pairs=semantic_pair_total,
        )
    per_predicate = []
    for predicate in sorted(predicate_totals):
        count = predicate_totals[predicate]
        name = (
            str(predicate_names[predicate])
            if predicate_names is not None and 0 <= predicate < len(predicate_names)
            else str(predicate)
        )
        per_predicate.append({
            "predicate_id": predicate,
            "predicate": name,
            "count": count,
            "hits": {str(k): predicate_hits[k][predicate] for k in topks},
            "recall": {
                str(k): predicate_hits[k][predicate] / max(count, 1) for k in topks
            },
        })
    return {
        "images": evaluated,
        "gt_pairs": total_gt,
        "base_pairs": base_pair_total,
        "semantic_pairs": semantic_pair_total,
        "semantic_keep_ratio": semantic_pair_total / max(base_pair_total, 1),
        "recall": {str(k): hits[k] / max(total_gt, 1) for k in topks},
        "hits": {str(k): hits[k] for k in topks},
        "avg_candidates": {
            str(k): candidate_totals[k] / max(evaluated, 1) for k in topks
        },
        "gt_relations": sum(predicate_totals.values()),
        "per_predicate": per_predicate,
        "seconds": time.perf_counter() - started,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    source = build_source(Path(args.data_root))
    records = build_records(source, args.split, args.limit)
    method = args.method.upper()
    checkpoint = {}
    if method == "PPN":
        model, checkpoint = load_pair_proposal_checkpoint(args.checkpoint, map_location="cpu")
        model.to(device)
        if list(source.ind_to_classes) != list(checkpoint.get("class_names", [])):
            raise ValueError("Checkpoint class order does not match the STAR dataset")
    elif method == "PPG":
        cfg = get_default_cfg()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        cfg["MODEL"]["BOX_MODE"] = "obb"
        cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"] = len(source.ind_to_classes)
        rel_cfg["PPG_ENABLED"] = True
        rel_cfg["TEST_FILTER_METHOD"] = "PPG"
        rel_cfg["PPG_MODEL_PATH_OBB"] = args.ppg_checkpoint
        rel_cfg["PPG_CHUNK_SIZE"] = args.block_size
        model = PairProposalGenerator(cfg).to(device).eval()
        if not model.loaded:
            raise RuntimeError(f"Failed to load PPG checkpoint: {args.ppg_checkpoint}")
    else:
        model = None
    semantic_support = None
    if not args.no_semantic_filter:
        semantic_support = load_semantic_pair_support(
            args.semantic_support, len(source.ind_to_classes)
        )
    metrics = evaluate_topk(
        model, records, device, args.topk, args.block_size, semantic_support,
        predicate_names=source.ind_to_predicates, method=method,
    )
    result = {
        "method": args.run_name or method,
        "filter_method": method,
        "checkpoint": str(args.checkpoint) if method == "PPN" else (
            str(args.ppg_checkpoint) if method == "PPG" else ""
        ),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "limit": args.limit,
        "semantic_filter": not args.no_semantic_filter,
        "semantic_support": None if args.no_semantic_filter else args.semantic_support,
        "metrics": metrics,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload, flush=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
