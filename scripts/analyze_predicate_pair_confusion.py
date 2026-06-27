from __future__ import annotations

import argparse
import ast
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    return values


def select_split_indices(split: str, split_mode: str, source_file: Path) -> List[int]:
    if split_mode != "fixed":
        raise ValueError("This script currently supports fixed split mode only.")
    return load_fixed_split_indices(source_file)[split]


def name_to_id(names: Sequence[str], target_name: str) -> int:
    for idx, name in enumerate(names):
        if name == target_name:
            return idx
    raise KeyError(f"Predicate not found: {target_name}")


def class_name(classes: Sequence[str], class_id: int) -> str:
    if 0 <= class_id < len(classes):
        return str(classes[class_id])
    return str(class_id)


def predicate_name(predicates: Sequence[str], pred_id: int) -> str:
    if 0 <= pred_id < len(predicates):
        return str(predicates[pred_id])
    return str(pred_id)


def load_split_relations(
    roidb_file: Path,
    split_indices: Iterable[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    records: Dict[int, Dict[str, np.ndarray]] = {}
    with h5py.File(roidb_file, "r") as roi_h5:
        labels_all = roi_h5["labels"][:, 0].astype(np.int64)
        im_to_first_box = roi_h5["img_to_first_box"][:]
        im_to_last_box = roi_h5["img_to_last_box"][:]
        im_to_first_rel = roi_h5["img_to_first_rel"][:]
        im_to_last_rel = roi_h5["img_to_last_rel"][:]
        rel_pairs = roi_h5["relationships"][:]
        rel_predicates = roi_h5["predicates"][:, 0]

        for image_index in split_indices:
            first_box = int(im_to_first_box[image_index])
            last_box = int(im_to_last_box[image_index])
            if first_box < 0 or last_box < first_box:
                continue

            labels_i = labels_all[first_box : last_box + 1].astype(np.int64)
            first_rel = int(im_to_first_rel[image_index])
            last_rel = int(im_to_last_rel[image_index])
            if first_rel >= 0 and last_rel >= first_rel:
                predicates_i = rel_predicates[first_rel : last_rel + 1].astype(np.int64)
                obj_idx = rel_pairs[first_rel : last_rel + 1] - first_box
                relations_i = np.column_stack((obj_idx, predicates_i)).astype(np.int64)
            else:
                relations_i = np.zeros((0, 3), dtype=np.int64)

            records[int(image_index)] = {
                "labels": labels_i,
                "relations": relations_i,
            }
    return records


def analyze_predicate_distribution(
    records: Dict[int, Dict[str, np.ndarray]],
    pred_id: int,
) -> Tuple[int, collections.Counter, collections.Counter]:
    rel_count = 0
    obj_pair_counter: collections.Counter = collections.Counter()
    image_counter: collections.Counter = collections.Counter()
    for image_id, record in records.items():
        labels = record["labels"]
        for subj, obj, pred in record["relations"].tolist():
            if int(pred) != pred_id:
                continue
            rel_count += 1
            pair = (int(labels[int(subj)]), int(labels[int(obj)]))
            obj_pair_counter[pair] += 1
            image_counter[image_id] += 1
    return rel_count, obj_pair_counter, image_counter


def analyze_cooccurrence_on_same_instance_pair(
    records: Dict[int, Dict[str, np.ndarray]],
    pred_a: int,
    pred_b: int,
) -> Tuple[int, int, List[Dict[str, object]]]:
    cooccur_pairs = 0
    cooccur_images = 0
    examples: List[Dict[str, object]] = []
    for image_id, record in records.items():
        grouped: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
        for subj, obj, pred in record["relations"].tolist():
            grouped[(int(subj), int(obj))].append(int(pred))
        image_hit = False
        for (subj, obj), preds in grouped.items():
            pred_set = set(preds)
            if pred_a in pred_set and pred_b in pred_set:
                cooccur_pairs += 1
                image_hit = True
                if len(examples) < 10:
                    labels = record["labels"]
                    examples.append(
                        {
                            "image_id": image_id,
                            "subj": subj,
                            "obj": obj,
                            "subj_label": int(labels[subj]),
                            "obj_label": int(labels[obj]),
                            "preds": sorted(pred_set),
                        }
                    )
        if image_hit:
            cooccur_images += 1
    return cooccur_pairs, cooccur_images, examples


def summarize_top_mistakes(
    mistake_json: Path,
    pred_a: str,
    pred_b: str,
) -> Dict[str, object]:
    data = json.loads(mistake_json.read_text(encoding="utf-8"))
    mistakes = data.get("top_mistakes_overall", [])
    a_to_b = None
    b_to_a = None
    outgoing = []
    incoming = []
    for item in mistakes:
        gt = item.get("gt_predicate")
        pd = item.get("predicted_predicate")
        if gt == pred_a and pd == pred_b:
            a_to_b = item
        if gt == pred_b and pd == pred_a:
            b_to_a = item
        if gt == pred_a or gt == pred_b:
            outgoing.append(item)
        if pd == pred_a or pd == pred_b:
            incoming.append(item)
    outgoing = sorted(outgoing, key=lambda x: int(x.get("count", 0)), reverse=True)[:10]
    incoming = sorted(incoming, key=lambda x: int(x.get("count", 0)), reverse=True)[:10]
    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "outgoing": outgoing,
        "incoming": incoming,
    }


def apply_runtime_cfg(cfg):
    import torch

    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def _find_pair_score(
    pred,
    subj: int,
    obj: int,
):
    import torch

    if not pred.has_field("rel_pair_idxs") or not pred.has_field("pred_rel_scores"):
        return None
    pair_idx = pred.get_field("rel_pair_idxs").long()
    rel_scores = pred.get_field("pred_rel_scores").float()
    if pair_idx.numel() == 0 or rel_scores.numel() == 0:
        return None
    valid = (pair_idx[:, 0] == int(subj)) & (pair_idx[:, 1] == int(obj))
    matched = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if matched.numel() == 0:
        return None
    if matched.numel() == 1:
        return rel_scores[int(matched.item())]
    selected = rel_scores[matched]
    best = selected[:, 1:].max(dim=1).values.argmax()
    return selected[int(best)]


def analyze_eval_logits(
    cfg: dict,
    checkpoint: str,
    pred_a_id: int,
    pred_b_id: int,
    pred_a_name: str,
    pred_b_name: str,
    topk_logits: int,
    max_examples: int,
    max_batches: int,
    device: str,
) -> None:
    import torch

    from sgg.data.build import build_dataloaders, build_datasets
    from sgg.engine import Trainer
    from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector

    apply_runtime_cfg(cfg)
    datasets = build_datasets(cfg, splits=("test",))
    test_meta = datasets["test"].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        test_meta.categories[i] for i in sorted(test_meta.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
        test_meta.predicates[i] for i in sorted(test_meta.predicates.keys())
    ]

    model = SceneGraphDetector(cfg)
    dataloaders = build_dataloaders(
        cfg,
        splits=("test",),
        datasets=datasets,
        shuffle_map={"test": False},
    )
    trainer = Trainer(cfg, model, device=device, dataloaders=dataloaders)
    trainer.load_checkpoint(checkpoint)
    loader = dataloaders["test"]
    model.eval()

    covered = 0
    missing_pair = 0
    total_gt_b = 0
    top1_counter: collections.Counter = collections.Counter()
    gt_rank_counter: collections.Counter = collections.Counter()
    same_first_diff_second = 0
    examples = []

    with torch.no_grad():
        for batch_idx, (images, targets, metas) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(trainer.device)
            targets = trainer._move_targets(targets)
            preds = model(images, targets if cfg["MODEL"]["TASK"] != "sgdet" else None)
            preds = [p.to("cpu") for p in preds]
            targets = [t.to("cpu") for t in targets]

            for pred, target, meta in zip(preds, targets, metas):
                gt_rels = target.get_field("relation_triplets").long() if target.has_field("relation_triplets") else torch.zeros((0, 3), dtype=torch.long)
                gt_labels = target.get_field("labels").long() if target.has_field("labels") else torch.zeros((0,), dtype=torch.long)
                for subj, obj, rel in gt_rels.tolist():
                    if int(rel) != pred_b_id:
                        continue
                    total_gt_b += 1
                    score = _find_pair_score(pred, int(subj), int(obj))
                    if score is None:
                        missing_pair += 1
                        continue
                    covered += 1
                    non_bg_scores = score[1:]
                    order = torch.argsort(non_bg_scores, descending=True)
                    top_ids = (order[:topk_logits] + 1).tolist()
                    top_scores = non_bg_scores[order[:topk_logits]].tolist()
                    top1 = int(top_ids[0]) if top_ids else -1
                    top1_counter[top1] += 1
                    gt_rank = int((order == (pred_b_id - 1)).nonzero(as_tuple=False)[0].item()) + 1
                    gt_rank_counter[gt_rank] += 1
                    if len(top_ids) >= 2 and top_ids[0] == pred_a_id and top_ids[1] == pred_b_id:
                        same_first_diff_second += 1
                    if len(examples) < max_examples:
                        examples.append(
                            {
                                "image_id": int(meta.get("image_id", -1)),
                                "subj": int(subj),
                                "obj": int(obj),
                                "subj_label": int(gt_labels[int(subj)].item()) if int(subj) < len(gt_labels) else -1,
                                "obj_label": int(gt_labels[int(obj)].item()) if int(obj) < len(gt_labels) else -1,
                                "top_ids": [int(x) for x in top_ids],
                                "top_scores": [float(x) for x in top_scores],
                                "gt_rank": gt_rank,
                            }
                        )

    print("\n=== TEST LOGIT RANKING FOR GT=B ===")
    print(f"checkpoint: {checkpoint}")
    print(f"target predicate B: {pred_b_name} (id={pred_b_id})")
    print(f"comparison predicate A: {pred_a_name} (id={pred_a_id})")
    print(f"total GT B relations: {total_gt_b}")
    print(f"covered by predicted pair set: {covered}")
    print(f"missing from predicted pair set: {missing_pair}")
    if covered > 0:
        print(f"A first and B second: {same_first_diff_second} / {covered} = {same_first_diff_second / covered:.6f}")

    print("\nTop-1 predicted predicate distribution on GT=B:")
    for pred_id, count in top1_counter.most_common(10):
        print(f"  {predicate_name(cfg['MODEL']['ROI_RELATION_HEAD']['RELATION_NAMES'], pred_id)} ({pred_id}): {count}")

    print("\nGT predicate rank distribution on covered GT=B pairs:")
    for rank, count in sorted(gt_rank_counter.items())[:10]:
        print(f"  rank {rank}: {count}")

    class_names = cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"]
    rel_names = cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"]
    if examples:
        print("\nExample covered GT=B pairs with top-k scores:")
        for ex in examples:
            top_pairs = [
                (int(pred_id), str(rel_names[int(pred_id)]), float(score))
                for pred_id, score in zip(ex["top_ids"], ex["top_scores"])
            ]
            print(
                f"  image_id={ex['image_id']} pair=({ex['subj']},{ex['obj']}) "
                f"{class_name(class_names, ex['subj_label'])}->{class_name(class_names, ex['obj_label'])} "
                f"gt_rank={ex['gt_rank']} topk={top_pairs}"
            )


def print_counter(title: str, counter: collections.Counter, classes: Sequence[str], limit: int = 10) -> None:
    print(f"\n{title}:")
    for (subj_cls, obj_cls), count in counter.most_common(limit):
        print(f"  {class_name(classes, subj_cls)} -> {class_name(classes, obj_cls)}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/star_predcls_obb_train.py")
    parser.add_argument("--mistakes-json", type=str, default="top_mistaken_predicates_test.json")
    parser.add_argument("--pred-a", type=str, default="parking in the same apron with")
    parser.add_argument("--pred-b", type=str, default="parking in the different apron with")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk-logits", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_py_config(args.config)
    dcfg_train = cfg["DATASETS"]["TRAIN"]
    dcfg_test = cfg["DATASETS"]["TEST"]
    roidb_file = Path(dcfg_train["ROIDB_FILE"])
    dict_file = Path(dcfg_train["DICT_FILE"])
    source_file = Path("sgg/data/sources/star.py")
    classes, predicates = load_star_info(dict_file)
    pred_a_id = name_to_id(predicates, args.pred_a)
    pred_b_id = name_to_id(predicates, args.pred_b)

    train_indices = select_split_indices("train", str(dcfg_train.get("SPLIT_MODE", "fixed")), source_file)
    test_indices = select_split_indices("test", str(dcfg_test.get("SPLIT_MODE", "fixed")), source_file)
    train_records = load_split_relations(roidb_file, train_indices)
    test_records = load_split_relations(roidb_file, test_indices)

    for split_name, records in (("train", train_records), ("test", test_records)):
        a_count, a_obj_pairs, _ = analyze_predicate_distribution(records, pred_a_id)
        b_count, b_obj_pairs, _ = analyze_predicate_distribution(records, pred_b_id)
        overlap_keys = set(a_obj_pairs) & set(b_obj_pairs)
        overlap_a = sum(a_obj_pairs[k] for k in overlap_keys)
        overlap_b = sum(b_obj_pairs[k] for k in overlap_keys)
        cooccur_pairs, cooccur_images, examples = analyze_cooccurrence_on_same_instance_pair(records, pred_a_id, pred_b_id)

        print(f"\n=== {split_name.upper()} ===")
        print(f"{args.pred_a} (id={pred_a_id}): {a_count}")
        print(f"{args.pred_b} (id={pred_b_id}): {b_count}")
        if b_count > 0:
            print(f"freq_ratio {args.pred_a} / {args.pred_b}: {a_count / b_count:.4f}")
        print(f"distinct subj-obj class pairs for A: {len(a_obj_pairs)}")
        print(f"distinct subj-obj class pairs for B: {len(b_obj_pairs)}")
        print(f"overlap subj-obj class pair types: {len(overlap_keys)}")
        print(f"A instances on overlapped class-pair types: {overlap_a}")
        print(f"B instances on overlapped class-pair types: {overlap_b}")
        print(f"same instance pair annotated with both A and B: {cooccur_pairs}")
        print(f"images containing same-pair A+B cooccurrence: {cooccur_images}")
        print_counter(f"Top subj-obj class pairs for {args.pred_a}", a_obj_pairs, classes)
        print_counter(f"Top subj-obj class pairs for {args.pred_b}", b_obj_pairs, classes)
        if examples:
            print("\nExample same-instance-pair A+B cooccurrence:")
            for ex in examples[:6]:
                pred_names = [predicate_name(predicates, pred_id) for pred_id in ex["preds"]]
                print(
                    f"  image_id={ex['image_id']} pair=({ex['subj']},{ex['obj']}) "
                    f"{class_name(classes, ex['subj_label'])}->{class_name(classes, ex['obj_label'])} "
                    f"preds={list(zip(ex['preds'], pred_names))}"
                )

    mistake_summary = summarize_top_mistakes(Path(args.mistakes_json), args.pred_a, args.pred_b)
    print("\n=== TOP MISTAKES ===")
    print(f"A -> B: {mistake_summary['a_to_b']}")
    print(f"B -> A: {mistake_summary['b_to_a']}")
    print("\nTop outgoing mistakes involving A/B:")
    for item in mistake_summary["outgoing"]:
        print(f"  {item}")
    print("\nTop incoming mistakes into A/B:")
    for item in mistake_summary["incoming"]:
        print(f"  {item}")

    if args.checkpoint:
        analyze_eval_logits(
            cfg=cfg,
            checkpoint=args.checkpoint,
            pred_a_id=pred_a_id,
            pred_b_id=pred_b_id,
            pred_a_name=args.pred_a,
            pred_b_name=args.pred_b,
            topk_logits=max(int(args.topk_logits), 1),
            max_examples=max(int(args.max_examples), 0),
            max_batches=int(args.max_batches),
            device=args.device,
        )


if __name__ == "__main__":
    main()
