"""Versioned, read-only proposal cache for the frozen STAR OBB detector.

Each file contains one image's post-patch/post-NMS detector output in the
1024px relation-view coordinate system.  GT labels and relations are excluded
on purpose.  ``CACHE_VERSION`` and the detector-config hash are correctness
barriers: a cache is reusable only when it would produce the same proposals as
the current raw detector path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from sgg.modeling.core.obb_ops import get_boxlist_angle_unit, set_boxlist_angle_unit
from sgg.structures.boxes import BoxList


CACHE_VERSION = 5
BOXES_PER_CLS_COORDINATE_FRAME = "relation_view"
_TENSOR_FIELDS = (
    "pred_labels",
    "pred_scores",
    "detector_nms_labels",
    "detector_nms_scores",
    "scores",
    "predict_logits",
    "pred_logits",
    "boxes_per_cls",
)


def original_sgdet_score_keep(
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    thresholds,
) -> torch.Tensor:
    """Return the final score selection used by the original STAR detector.

    ``two_stage_RS_multi_12.py::batch`` first keeps detections scoring at
    least 0.3.  It only falls back through 0.2/0.1/0.001/0.00001 when fewer
    than two object classes survive.  The source implementation stores
    detections class-by-class; counting unique labels is the equivalent
    operation after they have been flattened into a :class:`BoxList`.
    """
    if pred_scores.ndim != 1 or pred_labels.ndim != 1:
        raise ValueError("pred_scores and pred_labels must be one-dimensional")
    if pred_scores.numel() != pred_labels.numel():
        raise ValueError("pred_scores and pred_labels must have equal length")
    if pred_scores.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=pred_scores.device)

    fallback = None
    for threshold in thresholds:
        keep = torch.nonzero(pred_scores >= float(threshold), as_tuple=False).flatten()
        if keep.numel() == 0:
            continue
        fallback = keep
        if torch.unique(pred_labels[keep]).numel() >= 2:
            return keep
    if fallback is not None:
        return fallback
    # This can only occur when the configured threshold list is empty or all
    # scores lie below its last entry.  Retaining the detections matches the
    # source route's last-resort intent and avoids fabricating an empty image.
    return torch.arange(pred_scores.numel(), device=pred_scores.device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sgdet_detection_cache_payload(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("MODEL", {})
    sgdet_cfg = model_cfg.get("SGDET_COMPAT", {})
    test_cfg = cfg.get("TEST", {})
    return {
        "version": CACHE_VERSION,
        "pretrained_detector": model_cfg.get("PRETRAINED_DETECTOR", ""),
        "pretrained_detector_class_order": model_cfg.get(
            "PRETRAINED_DETECTOR_CLASS_ORDER", "background_first"
        ),
        "box_mode": model_cfg.get("BOX_MODE", ""),
        "obb_angle_unit": model_cfg.get("OBB_ANGLE_UNIT", ""),
        "sgdet": {
            "roi_bbox_target_stds": [0.1, 0.1, 0.2, 0.2, 0.1],
            "scale2_d1_refinement": True,
            "preserve_detector_nms_semantics": True,
            "use_d2": sgdet_cfg.get("USE_D2", True),
            "d2_scale": sgdet_cfg.get("D2_SCALE", 0.5),
            "rpn_anchor_offset": sgdet_cfg.get("RPN_ANCHOR_OFFSET", 0.0),
            "rpn_nms_pre": sgdet_cfg.get("RPN_NMS_PRE", 2000),
            "rpn_max_per_img": sgdet_cfg.get("RPN_MAX_PER_IMG", 2000),
            "rpn_nms_thresh": sgdet_cfg.get("RPN_NMS_THRESH", 0.8),
            "rcnn_score_thresh": sgdet_cfg.get("RCNN_SCORE_THRESH", 0.05),
            "rcnn_nms_thresh": sgdet_cfg.get("RCNN_NMS_THRESH", 0.1),
            "rcnn_max_per_img": sgdet_cfg.get("RCNN_MAX_PER_IMG", 2000),
            "patch_merge_nms_thresh": sgdet_cfg.get("PATCH_MERGE_NMS_THRESH", 0.4),
        },
        "patch": {
            "auto_enabled": test_cfg.get("PATCH_AUTO_ENABLED", True),
            "auto_min_size": test_cfg.get("PATCH_AUTO_MIN_SIZE", 1024),
            "max_pyramid_layers": test_cfg.get("PATCH_MAX_PYRAMID_LAYERS", 8),
            "batch_size": test_cfg.get("PATCH_BATCH_SIZE", 2),
            "batch_size_large": test_cfg.get("PATCH_BATCH_SIZE_LARGE", 4),
            "gaps": test_cfg.get("PATCH_GAPS", [200]),
            "size": test_cfg.get("PATCH_SIZE", [1024, 1024]),
            "score_thresholds": test_cfg.get(
                "PATCH_SCORE_THRESHOLDS", [0.3, 0.2, 0.1, 0.001, 0.00001]
            ),
        },
    }


def compute_sgdet_detection_cache_hash(cfg: Mapping[str, Any]) -> str:
    payload = _jsonable(sgdet_detection_cache_payload(cfg))
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve_sgdet_detection_cache_hash(cfg: Mapping[str, Any]) -> str:
    cache_cfg = (
        cfg.get("MODEL", {})
        .get("SGDET_COMPAT", {})
        .get("DETECTION_CACHE", {})
    )
    configured = str(cache_cfg.get("HASH", "") or "").strip()
    return configured or compute_sgdet_detection_cache_hash(cfg)


def sgdet_detection_cache_path(cache_dir: str | Path, split: str, image_id: int) -> Path:
    return Path(cache_dir) / str(split).lower() / f"{int(image_id):06d}.pt"


def serialize_sgdet_detection(
    proposal: BoxList,
    *,
    cfg: Mapping[str, Any],
    split: str,
    image_id: int,
    raw_size: Tuple[int, int],
    relation_size: Tuple[int, int],
) -> Dict[str, Any]:
    fields = {}
    for field in _TENSOR_FIELDS:
        if proposal.has_field(field):
            value = proposal.get_field(field)
            fields[field] = value.detach().cpu() if torch.is_tensor(value) else value
    if "pred_logits" not in fields and "predict_logits" in fields:
        fields["pred_logits"] = fields["predict_logits"]
    if "predict_logits" not in fields and "pred_logits" in fields:
        fields["predict_logits"] = fields["pred_logits"]
    return {
        "cache_version": CACHE_VERSION,
        "boxes_per_cls_coordinate_frame": BOXES_PER_CLS_COORDINATE_FRAME,
        "detector_config_hash": resolve_sgdet_detection_cache_hash(cfg),
        "detector_config_payload": _jsonable(sgdet_detection_cache_payload(cfg)),
        "image_id": int(image_id),
        "split": str(split).lower(),
        "raw_size": tuple(int(v) for v in raw_size),
        "relation_size": tuple(int(v) for v in relation_size),
        "bbox": proposal.bbox.detach().cpu(),
        "mode": proposal.mode,
        "angle_unit": get_boxlist_angle_unit(
            proposal, cfg.get("MODEL", {}).get("OBB_ANGLE_UNIT", "degree")
        ),
        "fields": fields,
    }


def load_sgdet_detection(
    path: str | Path,
    *,
    cfg: Mapping[str, Any],
    device: torch.device | str,
    expected_split: Optional[str] = None,
    expected_image_id: Optional[int] = None,
    expected_relation_size: Optional[Tuple[int, int]] = None,
) -> BoxList:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if int(payload.get("cache_version", -1)) != CACHE_VERSION:
        raise RuntimeError(f"Unsupported sgdet detection cache version in {path}: {payload.get('cache_version')}")
    coordinate_frame = str(payload.get("boxes_per_cls_coordinate_frame", ""))
    if coordinate_frame != BOXES_PER_CLS_COORDINATE_FRAME:
        raise RuntimeError(
            f"Invalid boxes_per_cls coordinate frame in {path}: "
            f"expected={BOXES_PER_CLS_COORDINATE_FRAME}, actual={coordinate_frame or '<missing>'}. "
            "Rebuild or migrate this sgdet detection cache."
        )
    expected_hash = resolve_sgdet_detection_cache_hash(cfg)
    actual_hash = str(payload.get("detector_config_hash", ""))
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"sgdet detection cache hash mismatch for {path}: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    if expected_split is not None and str(payload.get("split", "")).lower() != str(expected_split).lower():
        raise RuntimeError(
            f"sgdet detection cache split mismatch for {path}: "
            f"expected={expected_split}, actual={payload.get('split')}"
        )
    if expected_image_id is not None and int(payload.get("image_id", -1)) != int(expected_image_id):
        raise RuntimeError(
            f"sgdet detection cache image_id mismatch for {path}: "
            f"expected={expected_image_id}, actual={payload.get('image_id')}"
        )
    relation_size = tuple(int(v) for v in payload.get("relation_size", (0, 0)))
    if expected_relation_size is not None and relation_size != tuple(int(v) for v in expected_relation_size):
        raise RuntimeError(
            f"sgdet detection cache relation size mismatch for {path}: "
            f"expected={expected_relation_size}, actual={relation_size}"
        )
    box = BoxList(payload["bbox"].to(device), relation_size, str(payload.get("mode", "xywha")))
    angle_unit = payload.get("angle_unit", "")
    if box.mode == "xywha" and angle_unit:
        set_boxlist_angle_unit(box, str(angle_unit))
    for field, value in dict(payload.get("fields", {})).items():
        if torch.is_tensor(value):
            value = value.to(device)
        box.add_field(field, value)
    if box.has_field("predict_logits") and not box.has_field("pred_logits"):
        box.add_field("pred_logits", box.get_field("predict_logits"))
    if box.has_field("pred_logits") and not box.has_field("predict_logits"):
        box.add_field("predict_logits", box.get_field("pred_logits"))
    return box
