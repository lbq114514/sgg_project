from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.modeling.detectors.sgdet_detection_cache import (
    BOXES_PER_CLS_COORDINATE_FRAME,
    CACHE_VERSION,
    original_sgdet_score_keep,
    resolve_sgdet_detection_cache_hash,
    sgdet_detection_cache_payload,
)


def load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("sgdet_cache_migration_cfg", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if hasattr(module, "cfg"):
        return module.cfg
    if hasattr(module, "get_cfg"):
        return module.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def _assert_compatible_config(stored: dict, expected: dict, path: Path) -> None:
    stored = dict(stored or {})
    expected = dict(expected or {})
    # Cache schema migrations may alter only derived proposal fields.  All
    # detector settings that generated the proposals must remain identical.
    stored.pop("version", None)
    expected.pop("version", None)
    if stored != expected:
        raise RuntimeError(
            f"Detector config mismatch for {path}; refusing to rewrite a cache "
            "with settings different from the supplied config."
        )


def migrate_file(path: Path, *, cfg: dict, dry_run: bool) -> str:
    payload = torch.load(path, map_location="cpu")
    version = int(payload.get("cache_version", -1))
    frame = str(payload.get("boxes_per_cls_coordinate_frame", ""))
    if version == CACHE_VERSION and frame == BOXES_PER_CLS_COORDINATE_FRAME:
        return "skipped"
    if version < 5 and CACHE_VERSION >= 5:
        raise RuntimeError(
            f"Cache {path} predates the detector-fidelity fixes in cache v5 "
            "(RoI bbox target stds, scale=2 d1 refinement, and preserved "
            "detector NMS class/score semantics). These detections cannot be "
            "reconstructed from cached tensors; rebuild the cache."
        )
    if version == 2 and CACHE_VERSION >= 3:
        raise RuntimeError(
            f"Cache {path} is v2 and was built with the removed global "
            "post-merge RCNN_MAX_PER_IMG truncation. Missing detections cannot "
            "be reconstructed; rebuild this cache with the current detector code."
        )
    if version == 3 and CACHE_VERSION >= 4:
        expected_config = sgdet_detection_cache_payload(cfg)
        _assert_compatible_config(
            payload.get("detector_config_payload", {}),
            expected_config,
            path,
        )
        fields = dict(payload.get("fields", {}))
        scores = fields.get("pred_scores", fields.get("scores"))
        labels = fields.get("pred_labels")
        bbox = payload.get("bbox")
        if not torch.is_tensor(bbox) or not torch.is_tensor(scores) or not torch.is_tensor(labels):
            raise RuntimeError(f"Invalid v3 detection tensors in {path}")
        if bbox.size(0) != scores.numel() or bbox.size(0) != labels.numel():
            raise RuntimeError(f"Mismatched v3 detection tensor lengths in {path}")
        thresholds = (
            cfg.get("TEST", {}).get(
                "PATCH_SCORE_THRESHOLDS", [0.3, 0.2, 0.1, 0.001, 0.00001]
            )
        )
        keep = original_sgdet_score_keep(scores, labels, thresholds)
        if dry_run:
            return "would_migrate"
        payload["bbox"] = bbox[keep]
        for field, value in list(fields.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == bbox.size(0):
                fields[field] = value[keep]
        payload["fields"] = fields
        payload["cache_version"] = CACHE_VERSION
        payload["boxes_per_cls_coordinate_frame"] = BOXES_PER_CLS_COORDINATE_FRAME
        payload["detector_config_payload"] = expected_config
        payload["detector_config_hash"] = resolve_sgdet_detection_cache_hash(cfg)

        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return "migrated"
    if version != 1:
        raise RuntimeError(f"Unsupported source cache version in {path}: {version}")

    expected_config = sgdet_detection_cache_payload(cfg)
    _assert_compatible_config(
        payload.get("detector_config_payload", {}),
        expected_config,
        path,
    )
    bbox = payload.get("bbox")
    fields = dict(payload.get("fields", {}))
    boxes_per_cls = fields.get("boxes_per_cls")
    if not torch.is_tensor(bbox) or bbox.ndim != 2 or bbox.size(-1) != 5:
        raise RuntimeError(f"Invalid bbox tensor in {path}")
    if not torch.is_tensor(boxes_per_cls) or boxes_per_cls.ndim != 3:
        raise RuntimeError(f"Invalid boxes_per_cls tensor in {path}")
    if boxes_per_cls.size(0) != bbox.size(0) or boxes_per_cls.size(-1) != bbox.size(-1):
        raise RuntimeError(
            f"bbox/boxes_per_cls shape mismatch in {path}: "
            f"bbox={tuple(bbox.shape)}, boxes_per_cls={tuple(boxes_per_cls.shape)}"
        )
    if boxes_per_cls.numel() > 0:
        channel_delta = (boxes_per_cls - boxes_per_cls[:, :1, :]).abs().max().item()
        if channel_delta > 1e-5:
            raise RuntimeError(
                f"Cache {path} contains class-specific box regression "
                f"(max channel delta={channel_delta}); automatic migration is unsafe."
            )

    if dry_run:
        return "would_migrate"

    fields["boxes_per_cls"] = bbox[:, None, :].expand(
        -1, boxes_per_cls.size(1), -1
    ).clone()
    payload["fields"] = fields
    payload["cache_version"] = CACHE_VERSION
    payload["boxes_per_cls_coordinate_frame"] = BOXES_PER_CLS_COORDINATE_FRAME
    payload["detector_config_payload"] = expected_config
    payload["detector_config_hash"] = resolve_sgdet_detection_cache_hash(cfg)

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "migrated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate supported sgdet caches to the current schema. This "
            "repairs stale boxes_per_cls coordinates for v1 and applies the "
            "original detector's final score gate for recoverable schemas. "
            "Caches older than v5 must now be rebuilt because detector output "
            "semantics changed."
        )
    )
    parser.add_argument("--config", default="configs/star_sgdet_obb_train.py")
    parser.add_argument("--cache-dir", default="outputs/star_sgdet_detection_cache_v5")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_py_config(args.config)
    cache_dir = Path(args.cache_dir)
    splits = [item.strip().lower() for item in args.splits.split(",") if item.strip()]
    counts = {"migrated": 0, "would_migrate": 0, "skipped": 0}
    for split in splits:
        files = sorted((cache_dir / split).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No cache files found in {cache_dir / split}")
        for path in tqdm(files, desc=f"MigrateSgdetCache[{split}]"):
            status = migrate_file(path, cfg=cfg, dry_run=bool(args.dry_run))
            counts[status] += 1

        manifest_path = cache_dir / split / "manifest.json"
        if manifest_path.exists() and not args.dry_run:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_hash"] = resolve_sgdet_detection_cache_hash(cfg)
            manifest["computed_hash"] = resolve_sgdet_detection_cache_hash(cfg)
            manifest["config_payload"] = sgdet_detection_cache_payload(cfg)
            manifest["cache_version"] = CACHE_VERSION
            manifest["boxes_per_cls_coordinate_frame"] = BOXES_PER_CLS_COORDINATE_FRAME
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    print(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "splits": splits,
                "dry_run": bool(args.dry_run),
                "cache_version": CACHE_VERSION,
                "cache_hash": resolve_sgdet_detection_cache_hash(cfg),
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
