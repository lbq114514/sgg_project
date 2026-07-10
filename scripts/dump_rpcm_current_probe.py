#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rpcm_probe_utils import HookDumper, summarize_value, tensor_digest
from sgg.data.build import build_dataset
from sgg.data.collate import sgg_collate_fn
from sgg.engine.trainer import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from tools.eval_once import load_model_only_checkpoint


def load_cfg(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("probe_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.cfg


def find_dataset_index(dataset, image_id: int, image_file: str | None = None) -> int:
    """Resolve a probe sample by source filename when one is available.

    The SGG-Toolkit loader's ``image_id`` is its position in its own filtered
    dataset, whereas this project uses the STAR record index.  Those IDs are
    not interchangeable.  A filename is a stable cross-project identity.
    """
    if image_file:
        requested = Path(image_file).name
        matches = [
            index
            for index, record in enumerate(getattr(dataset, "records", ()))
            if Path(str(record.get("file_name", ""))).name == requested
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"image_file={requested!r} was not found in the selected dataset split")
        raise ValueError(f"image_file={requested!r} is ambiguous in the selected dataset split: {matches}")

    for idx in range(len(dataset)):
        _, _, meta = dataset[idx]
        if int(meta.get("image_id", -1)) == int(image_id):
            return idx
    raise ValueError(f"image_id={image_id} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one-image current RPCM intermediate tensors.")
    parser.add_argument("--config", default="configs/star_predcls_obb_train.py")
    parser.add_argument("--checkpoint", default="/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--sgcls-filter-label-source",
        default=None,
        choices=("gt", "pred"),
        help="Optional sgcls-only override for Semantic Filter and PPG/PPN/RSGP labels.",
    )
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Stop after detector object logits/features; do not execute the relation predictor.",
    )
    parser.add_argument("--image-id", type=int, default=4)
    parser.add_argument(
        "--image-file",
        default=None,
        help="Source image basename, preferred over --image-id for cross-project parity.",
    )
    parser.add_argument(
        "--reference-dump",
        default=None,
        help="Original probe dump; its image_filename is used to select the same current sample.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/rpcm_probe/current_image4.pt")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-cols", type=int, default=64)
    parser.add_argument("--max-flat", type=int, default=4096)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.sgcls_filter_label_source is not None:
        cfg["MODEL"]["ROI_RELATION_HEAD"]["SGCLS_FILTER_LABEL_SOURCE"] = args.sgcls_filter_label_source
    cfg["DATALOADER"]["NUM_WORKERS"] = 0
    dataset = build_dataset(cfg, args.split)
    meta = dataset.metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [meta.categories[i] for i in sorted(meta.categories)]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [meta.predicates[i] for i in sorted(meta.predicates)]
    image_file = args.image_file
    if args.reference_dump:
        reference_dump = torch.load(args.reference_dump, map_location="cpu")
        reference_image_file = reference_dump.get("image_filename")
        if not reference_image_file:
            raise ValueError(f"reference dump {args.reference_dump} has no image_filename")
        if image_file and Path(image_file).name != Path(reference_image_file).name:
            raise ValueError("--image-file and --reference-dump refer to different source files")
        image_file = reference_image_file
    idx = find_dataset_index(dataset, args.image_id, image_file=image_file)
    image, target, item_meta = dataset[idx]
    images, targets, metas = sgg_collate_fn(
        [(image, target, item_meta)],
        size_divisible=int(cfg["DATALOADER"].get("SIZE_DIVISIBLE", 0)),
    )
    device = torch.device(args.device)
    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=str(device))
    load_model_only_checkpoint(trainer, args.checkpoint, legacy_rpcm=True)
    model = trainer.model.eval()

    rel_head = model.roi_heads.relation
    predictor = rel_head.predictor
    hooks = HookDumper(max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat)
    hooks.add(model.backbone, "backbone")
    hooks.add(model.neck, "neck")
    hooks.add(model.roi_head.bbox_roi_extractor, "roi_head.bbox_roi_extractor")
    hooks.add(model.roi_head.bbox_head.shared_fcs[0], "roi_head.bbox_head.shared_fc0")
    hooks.add(model.roi_head.bbox_head.shared_fcs[1], "roi_head.bbox_head.shared_fc1")
    if not args.detector_only:
        hooks.add(rel_head.union_feature_extractor, "relation.union_feature_extractor")
        hooks.add(predictor.pairwise_feature_extractor, "predictor.pairwise_feature_extractor")
        hooks.add(predictor.down_samp, "predictor.down_samp")
        hooks.add(predictor.rel_residual, "predictor.rel_residual")
        hooks.add(predictor.rel_norm, "predictor.rel_norm")
        hooks.add(predictor.rel_proto, "predictor.rel_proto")

    with torch.inference_mode():
        device_targets = [target.to(device) for target in targets]
        if args.detector_only:
            features = model._extract_features(images.to(device))
            result = model._prepare_eval_proposals(features, images, device_targets)
        else:
            result = model(images.to(device), device_targets)
    hooks.close()
    pred = result[0].to("cpu")

    proposal_fields = {}
    for field in (
        "base_rel_pair_idxs",
        "sema_rel_pair_idxs",
        "final_rel_pair_idxs",
        "pruned_rel_pair_idxs",
        "pred_rel_scores",
        "pred_rel_labels",
        "rel_pair_idxs",
    ):
        if pred.has_field(field):
            proposal_fields[field] = summarize_value(
                pred.get_field(field),
                max_rows=args.max_rows,
                max_cols=args.max_cols,
                max_flat=args.max_flat,
            )
    object_fields = {}
    for field in (
        "labels",
        "gt_labels",
        "predict_logits",
        "pred_labels",
        "pred_scores",
        "filter_labels",
        "filter_label_source",
    ):
        if pred.has_field(field):
            object_fields[field] = summarize_value(
                pred.get_field(field),
                max_rows=args.max_rows,
                max_cols=args.max_cols,
                max_flat=args.max_flat,
            )

    target_cpu = targets[0]
    dump = {
        "source": "current",
        "config": str(Path(args.config).resolve()),
        "task": str(cfg["MODEL"].get("TASK", "")),
        "sgcls_filter_label_source": cfg["MODEL"]["ROI_RELATION_HEAD"].get(
            "SGCLS_FILTER_LABEL_SOURCE", "pred"
        ),
        "detector_only": bool(args.detector_only),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "image_id": int(args.image_id),
        "dataset_index": int(idx),
        "image_filename": str(item_meta.get("file_name", "")),
        "image": tensor_digest(images.cpu(), max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "target_size": tuple(target_cpu.size),
        "target_boxes": tensor_digest(target_cpu.bbox, max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "target_labels": tensor_digest(target_cpu.get_field("labels"), max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "target_relations": tensor_digest(target_cpu.get_field("relation_triplets"), max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "hooks": hooks.records,
        "object_fields": object_fields,
        "result_fields": proposal_fields,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
