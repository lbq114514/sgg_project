from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.data.sources.star import STARSource


DATA_ROOT = Path(
    os.environ.get(
        "STAR_SGG_ROOT",
        "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG",
    )
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build STAR semantic filter prior from the fixed train split.")
    parser.add_argument(
        "--image-root",
        default=str(DATA_ROOT / "STAR_img"),
        help="STAR image root.",
    )
    parser.add_argument(
        "--roidb-file",
        default=str(DATA_ROOT / "STAR-SGG-with-attri.h5"),
        help="STAR roidb h5 path.",
    )
    parser.add_argument(
        "--dict-file",
        default=str(DATA_ROOT / "STAR-SGG-dicts-with-attri.json"),
        help="STAR dictionary json path.",
    )
    parser.add_argument(
        "--image-file",
        default=str(DATA_ROOT / "STAR_image_data_v1.json"),
        help="STAR image metadata json path.",
    )
    parser.add_argument(
        "--current-sf",
        default="pretrained/SF_list.json",
        help="Existing SF json to compare against.",
    )
    parser.add_argument(
        "--output-sf",
        default="pretrained/SF_list_new.json",
        help="Output path for the rebuilt SF json.",
    )
    parser.add_argument(
        "--report-json",
        default="pretrained/SF_list_new_diff.json",
        help="Output path for the diff report json.",
    )
    parser.add_argument(
        "--split-mode",
        default="fixed",
        choices=["fixed", "random"],
        help="Dataset split mode used to construct the train set.",
    )
    parser.add_argument(
        "--num-im",
        type=int,
        default=-1,
        help="Optional image cap for debugging. -1 means all train images.",
    )
    parser.add_argument(
        "--topk-diff",
        type=int,
        default=50,
        help="How many largest-delta entries to include in the report.",
    )
    return parser.parse_args()


def build_sf_counts(source: STARSource, split_mode: str, num_im: int):
    records = source.get_split_records(
        split="train",
        box_mode="hbb",
        split_mode=split_mode,
        filter_empty_relations=False,
        filter_non_overlap=False,
        num_im=num_im,
    )
    num_classes = len(source.ind_to_classes)
    num_predicates = len(source.ind_to_predicates)
    counts = torch.zeros((num_classes, num_classes, num_predicates), dtype=torch.long)

    total_relations = 0
    images_with_relations = 0
    for record in records:
        labels = torch.as_tensor(record["labels"], dtype=torch.long)
        relations = torch.as_tensor(record["relations"], dtype=torch.long)
        if relations.numel() == 0:
            continue
        images_with_relations += 1
        total_relations += int(relations.size(0))
        subj = relations[:, 0]
        obj = relations[:, 1]
        pred = relations[:, 2]
        head_cls = labels[subj]
        tail_cls = labels[obj]
        valid = (
            (head_cls >= 0)
            & (head_cls < num_classes)
            & (tail_cls >= 0)
            & (tail_cls < num_classes)
            & (pred >= 0)
            & (pred < num_predicates)
        )
        if valid.any():
            flat_idx = (head_cls[valid] * num_classes + tail_cls[valid]) * num_predicates + pred[valid]
            counts.view(-1).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.long))

    return counts, records, total_relations, images_with_relations


def load_sf(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return torch.tensor(data, dtype=torch.long)


def summarize_diff(current: torch.Tensor, new: torch.Tensor, classes: list[str], predicates: list[str], topk: int):
    if current.shape != new.shape:
        raise ValueError(f"Shape mismatch: current {tuple(current.shape)} vs new {tuple(new.shape)}")

    current_nz = current > 0
    new_nz = new > 0
    diff = new - current
    changed = diff != 0
    support_only_current = current_nz & ~new_nz
    support_only_new = new_nz & ~current_nz
    support_overlap = current_nz & new_nz

    changed_idx = torch.nonzero(changed, as_tuple=False)
    if changed_idx.numel() > 0:
        abs_delta = diff.abs()[changed_idx[:, 0], changed_idx[:, 1], changed_idx[:, 2]]
        top_order = torch.argsort(abs_delta, descending=True)[:topk]
        top_entries = []
        for order_idx in top_order.tolist():
            h, t, r = changed_idx[order_idx].tolist()
            top_entries.append(
                {
                    "head_idx": h,
                    "tail_idx": t,
                    "predicate_idx": r,
                    "head_name": classes[h],
                    "tail_name": classes[t],
                    "predicate_name": predicates[r],
                    "current": int(current[h, t, r].item()),
                    "new": int(new[h, t, r].item()),
                    "delta": int(diff[h, t, r].item()),
                }
            )
    else:
        top_entries = []

    return {
        "shape": list(current.shape),
        "current_total_count": int(current.sum().item()),
        "new_total_count": int(new.sum().item()),
        "current_nonzero_entries": int(current_nz.sum().item()),
        "new_nonzero_entries": int(new_nz.sum().item()),
        "support_overlap_entries": int(support_overlap.sum().item()),
        "support_only_current_entries": int(support_only_current.sum().item()),
        "support_only_new_entries": int(support_only_new.sum().item()),
        "changed_entries": int(changed.sum().item()),
        "unchanged_entries": int((~changed).sum().item()),
        "abs_delta_sum": int(diff.abs().sum().item()),
        "top_abs_delta_entries": top_entries,
    }


def main():
    args = parse_args()

    source = STARSource.from_paths(
        image_root=Path(args.image_root),
        roidb_file=Path(args.roidb_file),
        dict_file=Path(args.dict_file),
        image_file=Path(args.image_file),
    )
    new_sf, records, total_relations, images_with_relations = build_sf_counts(
        source=source,
        split_mode=args.split_mode,
        num_im=args.num_im,
    )

    output_sf = Path(args.output_sf)
    output_sf.parent.mkdir(parents=True, exist_ok=True)
    output_sf.write_text(json.dumps(new_sf.tolist()), encoding="utf-8")

    current_sf = load_sf(Path(args.current_sf))
    report = summarize_diff(
        current=current_sf,
        new=new_sf,
        classes=source.ind_to_classes,
        predicates=source.ind_to_predicates,
        topk=args.topk_diff,
    )
    report.update(
        {
            "split": "train",
            "split_mode": args.split_mode,
            "num_images_used": len(records),
            "num_images_with_relations": images_with_relations,
            "num_relations_used": total_relations,
            "current_sf_path": str(Path(args.current_sf)),
            "output_sf_path": str(output_sf),
        }
    )

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built new SF from {len(records)} train images ({images_with_relations} with relations).")
    print(f"Total train relations counted: {total_relations}")
    print(f"Current SF total / nonzero: {report['current_total_count']} / {report['current_nonzero_entries']}")
    print(f"New SF total / nonzero: {report['new_total_count']} / {report['new_nonzero_entries']}")
    print(
        "Support overlap / only_current / only_new: "
        f"{report['support_overlap_entries']} / {report['support_only_current_entries']} / {report['support_only_new_entries']}"
    )
    print(f"Changed entries: {report['changed_entries']}, abs delta sum: {report['abs_delta_sum']}")
    print(f"Wrote new SF to {output_sf}")
    print(f"Wrote diff report to {report_path}")


if __name__ == "__main__":
    main()
