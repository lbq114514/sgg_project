"""Light wrapper around mmrotate OBB utility functions.

The STAR loader historically emitted OpenCV-style degree-valued OBB angles.
OpenMMLab's rotated operators in this environment use radians.  Keep explicit
angle-unit helpers here so callers can convert at operator boundaries instead
of silently mixing conventions.
"""

import math

import torch

try:
    from mmrotate.core import obb2poly as _mmrotate_obb2poly
    from mmrotate.core import obb2xyxy as _mmrotate_obb2xyxy
    from mmrotate.core import poly2obb as _mmrotate_poly2obb
except ImportError as e:
    raise ImportError(
        "Failed to import mmrotate core OBB functions. Please install mmrotate."
    ) from e


ANGLE_UNIT_FIELD = "box_angle_unit"
DEGREE = "degree"
RADIAN = "radian"


def normalize_angle_unit(unit) -> str:
    if unit is None:
        return DEGREE
    if torch.is_tensor(unit):
        if unit.numel() == 0:
            return DEGREE
        return RADIAN if int(unit.flatten()[0].item()) == 1 else DEGREE
    unit = str(unit).lower()
    if unit in {"deg", "degree", "degrees"}:
        return DEGREE
    if unit in {"rad", "radian", "radians"}:
        return RADIAN
    raise ValueError(f"Unsupported OBB angle unit: {unit!r}")


def get_boxlist_angle_unit(boxlist, default: str = DEGREE) -> str:
    if hasattr(boxlist, "has_field") and boxlist.has_field(ANGLE_UNIT_FIELD):
        return normalize_angle_unit(boxlist.get_field(ANGLE_UNIT_FIELD))
    return normalize_angle_unit(default)


def set_boxlist_angle_unit(boxlist, unit: str):
    boxlist.add_field(ANGLE_UNIT_FIELD, normalize_angle_unit(unit))
    return boxlist


def angle_to_radians(angle: torch.Tensor, angle_unit: str = DEGREE) -> torch.Tensor:
    return torch.deg2rad(angle) if normalize_angle_unit(angle_unit) == DEGREE else angle


def angle_from_radians(angle: torch.Tensor, angle_unit: str = DEGREE) -> torch.Tensor:
    return torch.rad2deg(angle) if normalize_angle_unit(angle_unit) == DEGREE else angle


def convert_obb_angle_unit(boxes: torch.Tensor, src_unit: str = DEGREE, dst_unit: str = RADIAN) -> torch.Tensor:
    src = normalize_angle_unit(src_unit)
    dst = normalize_angle_unit(dst_unit)
    if src == dst or boxes.numel() == 0:
        return boxes
    converted = boxes.clone()
    if converted.size(-1) < 5:
        return converted
    converted[..., 4] = angle_to_radians(converted[..., 4], src) if dst == RADIAN else angle_from_radians(converted[..., 4], dst)
    return converted


def _angle_period(angle_range: str, angle_unit: str):
    unit = normalize_angle_unit(angle_unit)
    if angle_range == "oc":
        return (math.pi / 2, 0.0) if unit == RADIAN else (90.0, 0.0)
    if angle_range == "le90":
        return (math.pi, -math.pi / 2) if unit == RADIAN else (180.0, -90.0)
    if angle_range == "le135":
        return (math.pi, -math.pi / 4) if unit == RADIAN else (180.0, -45.0)
    raise NotImplementedError(f"Unsupported angle_range: {angle_range}")


def norm_angle(angle, angle_range="oc", angle_unit: str = DEGREE):
    """
    Normalize rotated-box angles.

    ``angle_unit`` can be ``"degree"`` or ``"radian"``.
    """
    period, start = _angle_period(angle_range, angle_unit)

    if isinstance(angle, torch.Tensor):
        return torch.remainder(angle - start, period) + start
    return ((angle - start) % period) + start


def obb2xyxy(boxes: torch.Tensor, version: str = "oc", angle_unit: str = DEGREE) -> torch.Tensor:
    return _mmrotate_obb2xyxy(convert_obb_angle_unit(boxes, angle_unit, RADIAN), version)


def obb2poly(boxes: torch.Tensor, version: str = "oc", angle_unit: str = DEGREE) -> torch.Tensor:
    return _mmrotate_obb2poly(convert_obb_angle_unit(boxes, angle_unit, RADIAN), version)


def poly2obb(polys: torch.Tensor, version: str = "oc", angle_unit: str = DEGREE) -> torch.Tensor:
    boxes = _mmrotate_poly2obb(polys, version)
    return convert_obb_angle_unit(boxes, RADIAN, angle_unit)


__all__ = [
    "ANGLE_UNIT_FIELD",
    "DEGREE",
    "RADIAN",
    "angle_from_radians",
    "angle_to_radians",
    "convert_obb_angle_unit",
    "get_boxlist_angle_unit",
    "normalize_angle_unit",
    "norm_angle",
    "obb2poly",
    "obb2xyxy",
    "poly2obb",
    "set_boxlist_angle_unit",
]
