#!/usr/bin/env python3
"""Explicitly migrate a current-project checkpoint's detector class order."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.modeling.detectors.class_channel_order import (  # noqa: E402
    BACKGROUND_LAST,
    INTERNAL_DETECTOR_CLASS_ORDER,
    normalize_detector_class_order,
    reorder_detector_classifier_state_dict,
)


def _checkpoint_state_key(checkpoint: object) -> str:
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a current-project checkpoint dictionary, not a raw state_dict.")
    for key in ("model", "state_dict", "module"):
        if isinstance(checkpoint.get(key), dict):
            return key
    raise TypeError("Checkpoint does not contain a model/state_dict/module tensor mapping.")


def migrate_checkpoint(input_path: str | Path, output_path: str | Path, *, source_order: str) -> dict:
    """Write a model-only-safe, background-first checkpoint without mutating input."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("--output must differ from --input; the source checkpoint is never modified.")
    source_order = normalize_detector_class_order(source_order)
    if source_order == INTERNAL_DETECTOR_CLASS_ORDER:
        raise ValueError(
            "source-order is already background_first; refusing a no-op migration."
        )

    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a current-project checkpoint dictionary, not a raw state_dict.")
    if checkpoint.get("detector_class_channel_order") is not None:
        raise ValueError(
            "Checkpoint already declares detector_class_channel_order="
            f"{checkpoint['detector_class_channel_order']!r}; refusing to migrate twice."
        )
    state_key = _checkpoint_state_key(checkpoint)
    state_dict = checkpoint[state_key]
    changed = reorder_detector_classifier_state_dict(
        state_dict,
        source_order=source_order,
        target_order=INTERNAL_DETECTOR_CLASS_ORDER,
    )
    if not changed:
        raise ValueError(
            "No supported detector classifier tensors were found. Expected at least "
            "roi_head.bbox_head.fc_cls.weight/bias."
        )

    checkpoint["detector_class_channel_order"] = INTERNAL_DETECTOR_CLASS_ORDER
    checkpoint["detector_class_channel_order_migration"] = {
        "source_order": source_order,
        "target_order": INTERNAL_DETECTOR_CLASS_ORDER,
        "changed_keys": changed,
        "optimizer_scheduler_dropped": True,
    }
    # Optimizer buffers correspond to the pre-conversion classifier rows.  Do
    # not restore them accidentally when the migrated checkpoint is used.
    checkpoint["optimizer"] = None
    checkpoint["scheduler"] = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "source_order": source_order,
        "target_order": INTERNAL_DETECTOR_CLASS_ORDER,
        "changed_keys": changed,
        "optimizer_scheduler_dropped": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicitly migrate a current-project detector classifier from background-last to background-first."
    )
    parser.add_argument("--input", required=True, help="Existing current-project checkpoint.")
    parser.add_argument("--output", required=True, help="New migrated checkpoint path.")
    parser.add_argument(
        "--source-order",
        choices=(BACKGROUND_LAST,),
        default=BACKGROUND_LAST,
        help="Object-class channel order in the input checkpoint (default: background_last).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = migrate_checkpoint(args.input, args.output, source_order=args.source_order)
    print("Migrated detector class-channel order:", report, flush=True)


if __name__ == "__main__":
    main()
