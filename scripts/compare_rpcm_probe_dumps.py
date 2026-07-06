#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rpcm_probe_utils import compare_samples


def is_tensor_digest(value) -> bool:
    return isinstance(value, dict) and "shape" in value and "sample" in value


def flatten_digests(prefix: str, value, out: dict[str, dict]) -> None:
    if is_tensor_digest(value):
        out[prefix] = value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_digests(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            flatten_digests(f"{prefix}[{idx}]", child, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current/original RPCM probe dumps.")
    parser.add_argument("--left", default="outputs/rpcm_probe/original_image4.pt")
    parser.add_argument("--right", default="outputs/rpcm_probe/current_image4.pt")
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu")
    right = torch.load(args.right, map_location="cpu")
    print(f"left={left.get('source')} image_id={left.get('image_id')} file={args.left}")
    print(f"right={right.get('source')} image_id={right.get('image_id')} file={args.right}")

    left_flat: dict[str, dict] = {}
    right_flat: dict[str, dict] = {}
    for section in ("image", "target_boxes", "target_labels", "target_relations", "hooks", "result_fields"):
        flatten_digests(section, left.get(section), left_flat)
        flatten_digests(section, right.get(section), right_flat)

    common = sorted(set(left_flat) & set(right_flat))
    missing_left = sorted(set(right_flat) - set(left_flat))
    missing_right = sorted(set(left_flat) - set(right_flat))
    print(f"common={len(common)} missing_left={len(missing_left)} missing_right={len(missing_right)}")
    if missing_left:
        print("missing_left sample:", missing_left[:20])
    if missing_right:
        print("missing_right sample:", missing_right[:20])

    rows = []
    for key in common:
        comp = compare_samples(left_flat[key], right_flat[key])
        rows.append((key, comp))
    rows.sort(key=lambda item: float(item[1].get("sample_mean_abs", -1.0)), reverse=True)
    print("\nLargest sample differences:")
    for key, comp in rows[: args.top]:
        print(
            f"{key}: "
            f"shape {comp.get('left_shape')} vs {comp.get('right_shape')} | "
            f"mean {comp.get('left_mean'):.6g} vs {comp.get('right_mean'):.6g} | "
            f"std {comp.get('left_std'):.6g} vs {comp.get('right_std'):.6g} | "
            f"sample_max_abs={comp.get('sample_max_abs', 'NA')} "
            f"sample_mean_abs={comp.get('sample_mean_abs', 'NA')} "
            f"cos={comp.get('sample_cosine', 'NA')}"
        )

    print("\nLikely forward order:")
    for key, comp in sorted(rows, key=lambda item: item[0]):
        if key.startswith("hooks."):
            print(
                f"{key}: shape {comp.get('left_shape')} vs {comp.get('right_shape')}, "
                f"mean_abs={comp.get('sample_mean_abs', 'NA')}, cos={comp.get('sample_cosine', 'NA')}"
            )


if __name__ == "__main__":
    main()
