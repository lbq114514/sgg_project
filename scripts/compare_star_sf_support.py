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
    parser = argparse.ArgumentParser(description="Compare STAR support-only SF against the current SF.")
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
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--split-mode",
        default="fixed",
        choices=["fixed", "random"],
        help="Dataset split mode.",
    )
    parser.add_argument(
        "--num-im",
        type=int,
        default=-1,
        help="Optional image cap for debugging. -1 means all images from the split.",
    )
    parser.add_argument(
        "--current-sf",
        default="pretrained/SF_list.json",
        help="Existing SF json to compare against.",
    )
    parser.add_argument(
        "--output-support",
        default="pretrained/SF_list_support.json",
        help="Output path for the support-only SF json.",
    )
    parser.add_argument(
        "--report-json",
        default="pretrained/SF_list_support_diff.json",
        help="Output path for the support comparison report.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="How many support-difference entries to include in the report.",
    )
    return parser.parse_args()


def _load_records(source: STARSource, split: str, split_mode: str, num_im: int):
    if split != "all":
        return source.get_split_records(
            split=split,
            box_mode="hbb",
            split_mode=split_mode,
            filter_empty_relations=False,
            filter_non_overlap=False,
            num_im=num_im,
        )
    records = []
    for sub_split in ("train", "val", "test"):
        records.extend(
            source.get_split_records(
                split=sub_split,
                box_mode="hbb",
                split_mode=split_mode,
                filter_empty_relations=False,
                filter_non_overlap=False,
                num_im=-1,
            )
        )
    if num_im > -1:
        records = records[:num_im]
    return records


def build_support(source: STARSource, split: str, split_mode: str, num_im: int):
    records = _load_records(source, split=split, split_mode=split_mode, num_im=num_im)
    num_classes = len(source.ind_to_classes)
    num_predicates = len(source.ind_to_predicates)
    support = torch.zeros((num_classes, num_classes, num_predicates), dtype=torch.bool)

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
            support[head_cls[valid], tail_cls[valid], pred[valid]] = True

    return support, records, total_relations, images_with_relations


def load_sf_support(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return torch.tensor(data, dtype=torch.long) > 0


def support_report(current: torch.Tensor, new: torch.Tensor, classes: list[str], predicates: list[str], topk: int):
    if current.shape != new.shape:
        raise ValueError(f"Shape mismatch: current {tuple(current.shape)} vs new {tuple(new.shape)}")

    overlap = current & new
    only_current = current & ~new
    only_new = new & ~current
    union = current | new

    diff_idx = torch.nonzero(only_current | only_new, as_tuple=False)
    top_entries = []
    for h, t, r in diff_idx[:topk].tolist():
        top_entries.append(
            {
                "head_idx": h,
                "tail_idx": t,
                "predicate_idx": r,
                "head_name": classes[h],
                "tail_name": classes[t],
                "predicate_name": predicates[r],
                "in_current": bool(current[h, t, r].item()),
                "in_new": bool(new[h, t, r].item()),
            }
        )

    intersection = int(overlap.sum().item())
    current_count = int(current.sum().item())
    new_count = int(new.sum().item())
    union_count = int(union.sum().item())

    return {
        "shape": list(current.shape),
        "current_support_entries": current_count,
        "new_support_entries": new_count,
        "overlap_entries": intersection,
        "only_current_entries": int(only_current.sum().item()),
        "only_new_entries": int(only_new.sum().item()),
        "union_entries": union_count,
        "jaccard": float(intersection / max(union_count, 1)),
        "current_coverage_by_new": float(intersection / max(current_count, 1)),
        "new_coverage_by_current": float(intersection / max(new_count, 1)),
        "top_support_diff_entries": top_entries,
    }


def main():
    args = parse_args()
    source = STARSource.from_paths(
        image_root=Path(args.image_root),
        roidb_file=Path(args.roidb_file),
        dict_file=Path(args.dict_file),
        image_file=Path(args.image_file),
    )
    new_support, records, total_relations, images_with_relations = build_support(
        source=source,
        split=args.split,
        split_mode=args.split_mode,
        num_im=args.num_im,
    )

    output_support = Path(args.output_support)
    output_support.parent.mkdir(parents=True, exist_ok=True)
    output_support.write_text(json.dumps(new_support.long().tolist()), encoding="utf-8")

    current_support = load_sf_support(Path(args.current_sf))
    report = support_report(
        current=current_support,
        new=new_support,
        classes=source.ind_to_classes,
        predicates=source.ind_to_predicates,
        topk=args.topk,
    )
    report.update(
        {
            "split": args.split,
            "split_mode": args.split_mode,
            "num_images_used": len(records),
            "num_images_with_relations": images_with_relations,
            "num_relations_used": total_relations,
            "current_sf_path": str(Path(args.current_sf)),
            "output_support_path": str(output_support),
        }
    )

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built support-only SF from {len(records)} {args.split} images ({images_with_relations} with relations).")
    print(f"Total relations scanned: {total_relations}")
    print(
        "Current / new / overlap support entries: "
        f"{report['current_support_entries']} / {report['new_support_entries']} / {report['overlap_entries']}"
    )
    print(
        "Only current / only new / jaccard: "
        f"{report['only_current_entries']} / {report['only_new_entries']} / {report['jaccard']:.6f}"
    )
    print(f"Wrote support-only SF to {output_support}")
    print(f"Wrote support diff report to {report_path}")


if __name__ == "__main__":
    main()
