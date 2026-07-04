from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgg.modeling.roi_heads.pair_proposal_network import load_pair_proposal_checkpoint
from tools.train_pair_proposal_network import (
    DEFAULT_DATA_ROOT,
    build_records,
    build_source,
    load_semantic_pair_support,
    record_to_proposal,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Cache frozen-PPN ranked pairs for TypedHyperRPCM.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default="pretrained/PPN_OBB.pth")
    parser.add_argument("--semantic-support", default="pretrained/SF_list_support.json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=200000)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/pair_graph_cache/train_ppn_top10000.pth")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_pair_proposal_checkpoint(checkpoint_path, map_location=args.device)
    model.to(args.device).eval()
    num_classes = int(checkpoint["model_config"]["num_obj_classes"])
    allowed = load_semantic_pair_support(args.semantic_support, num_classes).to(args.device)
    records = build_records(build_source(Path(args.data_root)), args.split, args.limit)
    entries = {}
    for record in tqdm(records, desc=f"cache {args.split}"):
        proposal = record_to_proposal(record, torch.device(args.device))
        pairs, _ = model.topk_pairs(
            proposal, topk=args.topk, block_size=args.block_size,
            allowed_label_pairs=allowed,
        )
        entries[str(int(record["image_index"]))] = pairs.cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": {
            "split": args.split,
            "topk": args.topk,
            "block_size": args.block_size,
            "ppn_checkpoint": str(checkpoint_path),
            "ppn_sha256": file_sha256(checkpoint_path),
            "semantic_support": str(args.semantic_support),
        },
        "entries": entries,
    }, output)
    print({"output": str(output), "images": len(entries), "topk": args.topk})


if __name__ == "__main__":
    main()
