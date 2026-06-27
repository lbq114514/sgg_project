from __future__ import annotations

import argparse
import ast
import collections
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


def load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def load_star_info(dict_file: Path) -> Tuple[List[str], List[str]]:
    info = json.loads(dict_file.read_text(encoding="utf-8"))
    label_to_idx = dict(info["label_to_idx"])
    predicate_to_idx = dict(info["predicate_to_idx"])
    label_to_idx.setdefault("__background__", 0)
    predicate_to_idx.setdefault("__background__", 0)
    classes = sorted(label_to_idx, key=lambda key: label_to_idx[key])
    predicates = sorted(predicate_to_idx, key=lambda key: predicate_to_idx[key])
    return classes, predicates


def load_fixed_split_indices(source_file: Path) -> Dict[str, List[int]]:
    module_ast = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    values: Dict[str, List[int]] = {}
    wanted = {
        "DEFAULT_FIXED_TRAIN_INDEX": "train",
        "DEFAULT_FIXED_VAL_INDEX": "val",
        "DEFAULT_FIXED_TEST_INDEX": "test",
    }
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                values[wanted[target.id]] = ast.literal_eval(node.value)
    missing = sorted(set(wanted.values()) - set(values.keys()))
    if missing:
        raise KeyError(f"Missing fixed split indices for: {missing}")
    return values


def select_split_indices(
    split: str,
    num_images: int,
    split_mode: str,
    split_ratios: Sequence[int],
    random_seed: int,
    source_file: Path,
) -> List[int]:
    if split_mode == "fixed":
        return list(load_fixed_split_indices(source_file)[split])
    if split_mode == "random":
        indices = list(range(num_images))
        rng = random.Random(int(random_seed))
        rng.shuffle(indices)
        ratios = [int(v) for v in split_ratios]
        ratio_sum = max(sum(ratios), 1)
        n_train = int(round(num_images * ratios[0] / ratio_sum))
        n_val = int(round(num_images * ratios[1] / ratio_sum))
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        return {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def group_relations(relations: Iterable[Iterable[int]]) -> Dict[Tuple[int, int], List[int]]:
    grouped: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    for subj, obj, pred in relations:
        grouped[(int(subj), int(obj))].append(int(pred))
    return grouped


def build_relation_fields(
    relations: np.ndarray,
    num_objects: int,
    filter_duplicate_relations: bool,
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    rel_list = [[int(v) for v in rel] for rel in relations.tolist()]
    if not rel_list:
        return np.zeros((num_objects, num_objects), dtype=np.int64), []

    if filter_duplicate_relations:
        grouped: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        for subj, obj, pred in rel_list:
            grouped[(subj, obj)].append(pred)
        rel_list = [
            [subj, obj, rng.choice(preds)]
            for (subj, obj), preds in grouped.items()
        ]

    pair_labels = np.zeros((num_objects, num_objects), dtype=np.int64)
    kept: List[Tuple[int, int, int]] = []
    for subj, obj, pred in rel_list:
        if 0 <= subj < num_objects and 0 <= obj < num_objects and subj != obj:
            pair_labels[subj, obj] = pred
            kept.append((subj, obj, pred))
    return pair_labels, kept


def predicate_name(predicates: Sequence[str], pred_id: int) -> str:
    if 0 <= pred_id < len(predicates):
        return str(predicates[pred_id])
    return str(pred_id)


def class_name(classes: Sequence[str], class_id: int) -> str:
    if 0 <= class_id < len(classes):
        return str(classes[class_id])
    return str(class_id)


def analyze_split(cfg: dict, split: str, limit: int, topk_examples: int) -> None:
    dcfg = cfg["DATASETS"][split.upper()]
    roidb_file = Path(dcfg["ROIDB_FILE"])
    dict_file = Path(dcfg["DICT_FILE"])
    source_file = Path("sgg/data/sources/star.py")
    split_mode = str(dcfg.get("SPLIT_MODE", "fixed"))
    split_ratios = dcfg.get("SPLIT_RATIOS", (6, 2, 2))
    random_seed = int(dcfg.get("RANDOM_SEED", 42))
    filter_empty_relations = bool(dcfg.get("FILTER_EMPTY_RELATIONS", True))
    filter_duplicate_relations = bool(dcfg.get("FILTER_DUPLICATE_RELATIONS", True)) and split == "train"
    num_im = int(dcfg.get("NUM_IM", -1))
    classes, predicates = load_star_info(dict_file)

    with h5py.File(roidb_file, "r") as roi_h5:
        num_images = len(roi_h5["split"][:])
        labels_all = roi_h5["labels"][:, 0].astype(np.int64)
        im_to_first_box = roi_h5["img_to_first_box"][:]
        im_to_last_box = roi_h5["img_to_last_box"][:]
        im_to_first_rel = roi_h5["img_to_first_rel"][:]
        im_to_last_rel = roi_h5["img_to_last_rel"][:]
        rel_pairs = roi_h5["relationships"][:]
        rel_predicates = roi_h5["predicates"][:, 0]

        selected = select_split_indices(
            split=split,
            num_images=num_images,
            split_mode=split_mode,
            split_ratios=split_ratios,
            random_seed=random_seed,
            source_file=source_file,
        )
        if num_im > -1:
            selected = selected[:num_im]
        if limit > -1:
            selected = selected[:limit]

        rng = random.Random(random_seed)

        total_images = 0
        raw_rel_total = 0
        raw_pair_total = 0
        raw_multi_pair_total = 0
        raw_multi_distinct_pair_total = 0
        raw_extra_rel_total = 0
        target_triplet_total = 0
        target_pair_total = 0
        pair_label_total = 0
        images_with_multi_rel_pairs = 0
        images_with_multi_distinct_rel_pairs = 0
        images_changed_by_target_build = 0
        multi_pred_counter: collections.Counter[Tuple[int, ...]] = collections.Counter()
        overwrite_examples = []

        for image_index in selected:
            first_box = int(im_to_first_box[image_index])
            last_box = int(im_to_last_box[image_index])
            if first_box < 0 or last_box < first_box:
                continue

            first_rel = int(im_to_first_rel[image_index])
            last_rel = int(im_to_last_rel[image_index])
            if first_rel >= 0 and last_rel >= first_rel:
                predicates_i = rel_predicates[first_rel : last_rel + 1].astype(np.int64)
                obj_idx = rel_pairs[first_rel : last_rel + 1] - first_box
                relations_i = np.column_stack((obj_idx, predicates_i)).astype(np.int64)
            else:
                relations_i = np.zeros((0, 3), dtype=np.int64)

            if filter_empty_relations and len(relations_i) == 0:
                continue

            labels_i = labels_all[first_box : last_box + 1].astype(np.int64)
            pair_labels, rel_triplets = build_relation_fields(
                relations=relations_i,
                num_objects=len(labels_i),
                filter_duplicate_relations=filter_duplicate_relations,
                rng=rng,
            )

            raw_relations = relations_i.tolist()
            raw_grouped = group_relations(raw_relations)
            target_grouped = group_relations(rel_triplets)
            raw_multi_pairs = {pair: preds for pair, preds in raw_grouped.items() if len(preds) > 1}
            raw_multi_distinct_pairs = {
                pair: preds for pair, preds in raw_multi_pairs.items() if len(set(int(p) for p in preds)) > 1
            }

            total_images += 1
            raw_rel_total += len(raw_relations)
            raw_pair_total += len(raw_grouped)
            raw_multi_pair_total += len(raw_multi_pairs)
            raw_multi_distinct_pair_total += len(raw_multi_distinct_pairs)
            raw_extra_rel_total += sum(len(preds) - 1 for preds in raw_grouped.values())
            target_triplet_total += len(rel_triplets)
            target_pair_total += len(target_grouped)
            pair_label_total += int((pair_labels > 0).sum())

            if raw_multi_pairs:
                images_with_multi_rel_pairs += 1
                for preds in raw_multi_pairs.values():
                    multi_pred_counter[tuple(sorted(set(int(p) for p in preds)))] += 1
            if raw_multi_distinct_pairs:
                images_with_multi_distinct_rel_pairs += 1

            if raw_grouped != target_grouped:
                images_changed_by_target_build += 1

            if len(overwrite_examples) < topk_examples:
                for (subj, obj), preds in raw_multi_distinct_pairs.items():
                    overwrite_examples.append(
                        {
                            "image_id": int(image_index),
                            "subj": int(subj),
                            "obj": int(obj),
                            "subj_label": int(labels_i[subj]),
                            "obj_label": int(labels_i[obj]),
                            "raw_preds": [int(p) for p in preds],
                            "kept_pred": int(pair_labels[subj, obj]),
                        }
                    )
                    if len(overwrite_examples) >= topk_examples:
                        break

    print(f"Split: {split}")
    print(f"Images checked: {total_images}")
    print(f"Filter duplicate relations: {filter_duplicate_relations}")
    print(f"Raw relations: {raw_rel_total}")
    print(f"Raw unique pairs: {raw_pair_total}")
    print(f"Raw multi-rel pairs: {raw_multi_pair_total}")
    print(f"Raw multi-rel pairs with distinct predicates: {raw_multi_distinct_pair_total}")
    print(f"Raw extra relations beyond 1-per-pair: {raw_extra_rel_total}")
    print(f"Target relation_triplets: {target_triplet_total}")
    print(f"Target unique triplet pairs: {target_pair_total}")
    print(f"Target pair_labels positive pairs: {pair_label_total}")
    print(f"Images with raw multi-rel pairs: {images_with_multi_rel_pairs}")
    print(f"Images with raw multi-rel pairs carrying distinct predicates: {images_with_multi_distinct_rel_pairs}")
    print(f"Images changed by target build: {images_changed_by_target_build}")
    if raw_rel_total > 0:
        print(f"Collapse ratio from raw multi-rel pairs: {raw_extra_rel_total / raw_rel_total:.6f}")

    top_patterns = multi_pred_counter.most_common(10)
    if top_patterns:
        print("\nTop multi-predicate pair patterns:")
        for pred_tuple, count in top_patterns:
            pred_names = [predicate_name(predicates, pred_id) for pred_id in pred_tuple]
            print(f"  count={count:5d} preds={list(pred_tuple)} names={pred_names}")

    if overwrite_examples:
        print("\nExample overwritten pairs:")
        for ex in overwrite_examples:
            subj_name = class_name(classes, ex["subj_label"])
            obj_name = class_name(classes, ex["obj_label"])
            raw_names = [predicate_name(predicates, pred) for pred in ex["raw_preds"]]
            kept_name = predicate_name(predicates, ex["kept_pred"])
            print(
                f"  image_id={ex['image_id']} pair=({ex['subj']},{ex['obj']}) "
                f"{subj_name}->{obj_name} "
                f"raw={list(zip(ex['raw_preds'], raw_names))} "
                f"kept={ex['kept_pred']}:{kept_name}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/star_predcls_obb_train.py")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--topk-examples", type=int, default=10)
    args = parser.parse_args()

    cfg = load_py_config(args.config)
    analyze_split(cfg, split=args.split, limit=args.limit, topk_examples=args.topk_examples)


if __name__ == "__main__":
    main()
