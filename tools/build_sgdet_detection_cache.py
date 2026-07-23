from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.data.build import build_dataloaders, build_datasets
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.modeling.detectors.sgdet_detection_cache import (
    compute_sgdet_detection_cache_hash,
    resolve_sgdet_detection_cache_hash,
    serialize_sgdet_detection,
    sgdet_detection_cache_path,
    sgdet_detection_cache_payload,
)


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


def parse_splits(value: str) -> list[str]:
    splits = []
    for item in value.replace(";", ",").split(","):
        item = item.strip().lower()
        if item:
            splits.append(item)
    if not splits:
        raise ValueError("--split must contain at least one split")
    return splits


def image_id_from_target_or_meta(target, meta: dict) -> int:
    image_id = meta.get("image_id")
    if image_id is None and target.has_field("image_id"):
        tensor = target.get_field("image_id")
        if torch.is_tensor(tensor) and tensor.numel() > 0:
            image_id = int(tensor.reshape(-1)[0].item())
    if image_id is None:
        raise RuntimeError("Cannot build sgdet detection cache without image_id")
    return int(image_id)


def raw_size_from_meta_or_target(meta: dict, detector_target) -> tuple[int, int]:
    if detector_target is not None:
        if hasattr(detector_target, "size"):
            return tuple(int(v) for v in detector_target.size)
        return tuple(int(v) for v in detector_target)
    if "sgdet_detector_size" in meta:
        return tuple(int(v) for v in meta["sgdet_detector_size"])
    return int(meta["width"]), int(meta["height"])


def prepare_cfg(args) -> dict:
    cfg = load_py_config(args.config)
    cfg["MODEL"]["TASK"] = "sgdet"
    cfg["MODEL"].setdefault("SGDET_COMPAT", {})
    cfg["MODEL"]["SGDET_COMPAT"].setdefault("DETECTION_CACHE", {})
    cfg["MODEL"]["SGDET_COMPAT"]["DETECTION_CACHE"]["ENABLED"] = False
    if args.num_workers is not None:
        cfg["DATALOADER"]["NUM_WORKERS"] = int(args.num_workers)
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
        cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = batch_size
        cfg["DATALOADER"]["VAL_BATCH_SIZE"] = batch_size
        cfg["DATALOADER"]["TEST_BATCH_SIZE"] = batch_size
        cfg["SOLVER"]["IMS_PER_BATCH"] = batch_size
        cfg["TEST"]["IMS_PER_BATCH"] = batch_size
    for split in ("TRAIN", "VAL", "TEST"):
        if split in cfg.get("DATASETS", {}):
            cfg["DATASETS"][split]["SGDET_DETECTOR_ORIGINAL_RESOLUTION"] = True
            cfg["DATASETS"][split]["AUGMENT"] = False
    return cfg


@torch.no_grad()
def build_split_cache(
    *,
    cfg: dict,
    split: str,
    output_dir: Path,
    device: str,
    max_images: int,
    overwrite: bool,
) -> dict:
    datasets = build_datasets(cfg, splits=(split,))
    dataset = datasets[split]
    metadata = dataset.metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        metadata.categories[i] for i in sorted(metadata.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
        metadata.predicates[i] for i in sorted(metadata.predicates.keys())
    ]
    dataloaders = build_dataloaders(
        cfg,
        splits=(split,),
        datasets=datasets,
        shuffle_map={split: False},
    )
    loader = dataloaders[split]
    model = SceneGraphDetector(cfg).to(device)
    model.eval()
    trainer = Trainer(cfg, model, device=device, dataloaders={})

    cache_hash = resolve_sgdet_detection_cache_hash(cfg)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    processed = 0

    for _, targets, metas in tqdm(loader, desc=f"BuildSgdetCache[{split}]"):
        if max_images > 0 and processed >= max_images:
            break
        detector_images, detector_targets = trainer._sgdet_detector_inputs_from_metas(metas)
        if detector_images is None:
            raise RuntimeError(
                "sgdet detector original-resolution stream is missing; "
                "check DATASETS.*.SGDET_DETECTOR_ORIGINAL_RESOLUTION"
            )
        batch_count = len(targets)
        if max_images > 0:
            keep = max(0, min(batch_count, max_images - processed))
            targets = targets[:keep]
            metas = metas[:keep]
            detector_images = detector_images[:keep]
            detector_targets = detector_targets[:keep]
            batch_count = keep
            if batch_count == 0:
                break
        proposals = model._detect_sgdet_raw_to_relation_view(
            detector_images,
            detector_targets,
            targets,
        )
        for target, meta, detector_target, proposal in zip(targets, metas, detector_targets, proposals):
            image_id = image_id_from_target_or_meta(target, meta)
            cache_path = sgdet_detection_cache_path(output_dir, split, image_id)
            if cache_path.exists() and not overwrite:
                skipped += 1
                continue
            payload = serialize_sgdet_detection(
                proposal,
                cfg=cfg,
                split=split,
                image_id=image_id,
                raw_size=raw_size_from_meta_or_target(meta, detector_target),
                relation_size=tuple(int(v) for v in target.size),
            )
            torch.save(payload, cache_path)
            written += 1
        processed += batch_count

    manifest = {
        "split": split,
        "cache_hash": cache_hash,
        "computed_hash": compute_sgdet_detection_cache_hash(cfg),
        "config_payload": sgdet_detection_cache_payload(cfg),
        "processed": processed,
        "written": written,
        "skipped": skipped,
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build frozen sgdet detector proposal cache.")
    parser.add_argument("--config", default="configs/star_sgdet_obb_train.py")
    parser.add_argument("--split", default="train", help="Comma-separated splits, e.g. train,test")
    parser.add_argument("--output-dir", default="outputs/star_sgdet_detection_cache_v5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = prepare_cfg(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in parse_splits(args.split):
        summaries.append(
            build_split_cache(
                cfg=cfg,
                split=split,
                output_dir=output_dir,
                device=args.device,
                max_images=int(args.max_images),
                overwrite=bool(args.overwrite),
            )
        )
    print(json.dumps({"output_dir": str(output_dir), "splits": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
