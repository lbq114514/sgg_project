"""Compatibility helpers for detector object-class channel conventions.

The original STAR OBB detector stores its background classifier row at the
last index.  This project uses the common SGG convention where background is
row zero.  Only the detector's object classification linear layers use this
conversion; relation-predicate logits are intentionally out of scope.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import List

import torch


BACKGROUND_FIRST = "background_first"
BACKGROUND_LAST = "background_last"
INTERNAL_DETECTOR_CLASS_ORDER = BACKGROUND_FIRST

DETECTOR_CLASSIFIER_KEYS = frozenset(
    {
        "roi_head.bbox_head.fc_cls.weight",
        "roi_head.bbox_head.fc_cls.bias",
        "roi_head_d2.bbox_head.fc_cls.weight",
        "roi_head_d2.bbox_head.fc_cls.bias",
    }
)


def normalize_detector_class_order(order: str) -> str:
    """Validate and normalize a detector object-class channel convention."""
    normalized = str(order).strip().lower()
    valid = {BACKGROUND_FIRST, BACKGROUND_LAST}
    if normalized not in valid:
        raise ValueError(
            "Unsupported detector class-channel order "
            f"{order!r}; expected one of {sorted(valid)}."
        )
    return normalized


def is_detector_classifier_key(key: str) -> bool:
    """Return whether ``key`` is one of the four supported detector heads."""
    return str(key) in DETECTOR_CLASSIFIER_KEYS


def reorder_detector_classifier_rows(
    value: torch.Tensor,
    *,
    source_order: str,
    target_order: str = INTERNAL_DETECTOR_CLASS_ORDER,
) -> torch.Tensor:
    """Return a classifier weight/bias tensor converted between BG conventions.

    The tensor is never mutated.  For a source with background at the last
    row, ``[bg, cls_1, ..., cls_n]`` becomes ``[last, 0, ..., last - 1]``.
    The same operation works for both weight matrices and bias vectors.
    """
    source_order = normalize_detector_class_order(source_order)
    target_order = normalize_detector_class_order(target_order)
    if source_order == target_order:
        return value
    if not torch.is_tensor(value) or value.ndim < 1 or value.size(0) < 2:
        raise ValueError(
            "Detector classifier tensors must have at least two rows; "
            f"received shape={tuple(value.shape) if torch.is_tensor(value) else type(value)!r}."
        )

    num_classes = value.size(0)
    if source_order == BACKGROUND_LAST and target_order == BACKGROUND_FIRST:
        permutation = torch.cat(
            (
                torch.tensor([num_classes - 1], device=value.device),
                torch.arange(num_classes - 1, device=value.device),
            )
        )
    elif source_order == BACKGROUND_FIRST and target_order == BACKGROUND_LAST:
        permutation = torch.cat(
            (
                torch.arange(1, num_classes, device=value.device),
                torch.tensor([0], device=value.device),
            )
        )
    else:  # normalize_detector_class_order already constrains this branch.
        raise AssertionError(f"Unhandled class-channel conversion: {source_order} -> {target_order}")
    return value.index_select(0, permutation)


def reorder_detector_classifier_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    source_order: str,
    target_order: str = INTERNAL_DETECTOR_CLASS_ORDER,
) -> List[str]:
    """In-place conversion of supported detector classifier tensors.

    Returns the exact keys that were changed.  Missing d2 keys are allowed so
    older checkpoints without the preserved d2 branch remain migratable.
    """
    source_order = normalize_detector_class_order(source_order)
    target_order = normalize_detector_class_order(target_order)
    if source_order == target_order:
        return []

    changed = []
    for key in sorted(DETECTOR_CLASSIFIER_KEYS):
        if key not in state_dict:
            continue
        state_dict[key] = reorder_detector_classifier_rows(
            state_dict[key],
            source_order=source_order,
            target_order=target_order,
        )
        changed.append(key)
    return changed
