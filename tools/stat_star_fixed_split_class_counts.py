#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "STAR_SGG_ROOT",
        "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG",
    )
)


def load_fixed_split_indices() -> Tuple[List[int], List[int], List[int]]:
    source_file = PROJECT_ROOT / "sgg/data/sources/star.py"
    module = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    values: Dict[str, List[int]] = {}
    target_names = {
        "DEFAULT_FIXED_TRAIN_INDEX",
        "DEFAULT_FIXED_VAL_INDEX",
        "DEFAULT_FIXED_TEST_INDEX",
    }
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in target_names:
                values[target.id] = list(ast.literal_eval(node.value))
    missing = target_names.difference(values)
    if missing:
        raise RuntimeError(f"Failed to load fixed split indices from {source_file}: missing {sorted(missing)}")
    return (
        values["DEFAULT_FIXED_TRAIN_INDEX"],
        values["DEFAULT_FIXED_VAL_INDEX"],
        values["DEFAULT_FIXED_TEST_INDEX"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count STAR object and relation class frequencies under the fixed split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"STAR dataset root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--objects-file",
        type=Path,
        default=None,
        help="Path to STAR_objects_v1.json. Overrides --data-root when provided.",
    )
    parser.add_argument(
        "--relationships-file",
        type=Path,
        default=None,
        help="Path to STAR_relationships_v1.json. Overrides --data-root when provided.",
    )
    parser.add_argument(
        "--dict-file",
        type=Path,
        default=None,
        help="Path to STAR-SGG-dicts-with-attri.json. Overrides --data-root when provided.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which fixed split to count. Default: all.",
    )
    parser.add_argument(
        "--filter-empty-relations",
        action="store_true",
        help="Match training dataset behavior and skip images without relations.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    return parser.parse_args()


def load_vocab(dict_file: Path) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, int]]:
    info = json.loads(dict_file.read_text(encoding="utf-8"))
    label_to_idx = {k: int(v) for k, v in info["label_to_idx"].items()}
    predicate_to_idx = {k: int(v) for k, v in info["predicate_to_idx"].items()}
    labels = [name for name, _ in sorted(label_to_idx.items(), key=lambda item: item[1])]
    predicates = [name for name, _ in sorted(predicate_to_idx.items(), key=lambda item: item[1])]
    return labels, predicates, label_to_idx, predicate_to_idx


def get_split_indices(split: str, num_images: int) -> Dict[str, List[int]]:
    train_idx, val_idx, test_idx = load_fixed_split_indices()
    split_to_indices = {
        "train": [idx for idx in train_idx if 0 <= idx < num_images],
        "val": [idx for idx in val_idx if 0 <= idx < num_images],
        "test": [idx for idx in test_idx if 0 <= idx < num_images],
    }
    if split == "all":
        return split_to_indices
    return {split: split_to_indices[split]}


def count_split(
    objects_by_image: Dict[int, List[Dict[str, object]]],
    relations_by_image: Dict[int, List[Dict[str, object]]],
    image_indices: Sequence[int],
    filter_empty_relations: bool,
    label_to_idx: Dict[str, int],
    predicate_to_idx: Dict[str, int],
) -> Dict[str, object]:
    obj_counter: Counter[int] = Counter()
    rel_counter: Counter[int] = Counter()
    num_images_counted = 0
    num_images_skipped = 0
    unknown_object_names: Counter[str] = Counter()
    unknown_predicate_names: Counter[str] = Counter()

    for image_index in image_indices:
        objects = objects_by_image.get(image_index, [])
        relations = relations_by_image.get(image_index, [])
        if not objects:
            num_images_skipped += 1
            continue
        if filter_empty_relations and not relations:
            num_images_skipped += 1
            continue

        for obj in objects:
            name = normalize_object_name(obj.get("names"))
            class_idx = label_to_idx.get(name)
            if class_idx is None:
                unknown_object_names[name] += 1
                continue
            obj_counter[class_idx] += 1

        for rel in relations:
            predicate = str(rel.get("predicate", "")).strip()
            class_idx = predicate_to_idx.get(predicate)
            if class_idx is None:
                unknown_predicate_names[predicate] += 1
                continue
            rel_counter[class_idx] += 1

        num_images_counted += 1

    return {
        "images_selected": len(image_indices),
        "images_counted": num_images_counted,
        "images_skipped": num_images_skipped,
        "object_counts": obj_counter,
        "relation_counts": rel_counter,
        "unknown_object_names": unknown_object_names,
        "unknown_predicate_names": unknown_predicate_names,
    }


def normalize_object_name(value: object) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    return str(value).strip()


def load_annotations_by_image(path: Path, key: str) -> Dict[int, List[Dict[str, object]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    annotations: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        annotations[image_id] = list(row.get(key, []))
    return annotations


def format_counts(names: Sequence[str], counts: Counter[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for class_idx, class_name in enumerate(names, start=1):
        rows.append(
            {
                "class_id": class_idx,
                "class_name": class_name,
                "count": int(counts.get(class_idx, 0)),
            }
        )
    return rows


def print_text_report(split_results: Dict[str, Dict[str, object]], obj_names: Sequence[str], rel_names: Sequence[str]) -> None:
    for split_name, result in split_results.items():
        print(f"=== {split_name} ===")
        print(
            "images_selected={images_selected}, images_counted={images_counted}, images_skipped={images_skipped}".format(
                **result
            )
        )
        if result["unknown_object_names"]:
            print(f"unknown_object_names={dict(result['unknown_object_names'])}")
        if result["unknown_predicate_names"]:
            print(f"unknown_predicate_names={dict(result['unknown_predicate_names'])}")
        print("object_class_counts")
        for row in format_counts(obj_names, result["object_counts"]):
            print(f"{row['class_id']:>3}\t{row['class_name']}\t{row['count']}")
        print("relation_class_counts")
        for row in format_counts(rel_names, result["relation_counts"]):
            print(f"{row['class_id']:>3}\t{row['class_name']}\t{row['count']}")
        print()


def build_json_report(split_results: Dict[str, Dict[str, object]], obj_names: Sequence[str], rel_names: Sequence[str]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for split_name, result in split_results.items():
        payload[split_name] = {
            "images_selected": int(result["images_selected"]),
            "images_counted": int(result["images_counted"]),
            "images_skipped": int(result["images_skipped"]),
            "unknown_object_names": dict(result["unknown_object_names"]),
            "unknown_predicate_names": dict(result["unknown_predicate_names"]),
            "object_class_counts": format_counts(obj_names, result["object_counts"]),
            "relation_class_counts": format_counts(rel_names, result["relation_counts"]),
        }
    return payload


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    objects_file = args.objects_file or (data_root / "STAR_objects_v1.json")
    relationships_file = args.relationships_file or (data_root / "STAR_relationships_v1.json")
    dict_file = args.dict_file or (data_root / "STAR-SGG-dicts-with-attri.json")

    obj_names, rel_names, label_to_idx, predicate_to_idx = load_vocab(dict_file)
    objects_by_image = load_annotations_by_image(objects_file, key="objects")
    relations_by_image = load_annotations_by_image(relationships_file, key="relationships")
    num_images = max(
        max(objects_by_image.keys(), default=-1),
        max(relations_by_image.keys(), default=-1),
    ) + 1

    split_to_indices = get_split_indices(args.split, num_images=num_images)
    split_results = {
        split_name: count_split(
            objects_by_image=objects_by_image,
            relations_by_image=relations_by_image,
            image_indices=image_indices,
            filter_empty_relations=args.filter_empty_relations,
            label_to_idx=label_to_idx,
            predicate_to_idx=predicate_to_idx,
        )
        for split_name, image_indices in split_to_indices.items()
    }

    if args.as_json:
        print(json.dumps(build_json_report(split_results, obj_names, rel_names), ensure_ascii=False, indent=2))
        return

    print_text_report(split_results, obj_names, rel_names)


if __name__ == "__main__":
    main()
