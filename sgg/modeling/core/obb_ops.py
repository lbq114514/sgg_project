"""Dependency-light OBB geometry helpers.

The STAR loader historically emitted OpenCV-style degree-valued OBB angles.
OpenMMLab's rotated operators in this environment use radians.  Keep explicit
angle-unit helpers here so callers can convert at operator boundaries instead
of silently mixing conventions.

The conversion routines intentionally reproduce the tensor behavior used by
``mmrotate.core.bbox.transforms``.  Keeping these three small operations local
avoids importing the complete mmrotate/mmdet detector stack just to convert
between OBB, polygon, and HBB representations.  Compiled rotated operators
such as rotated IoU, NMS, and RoIAlign still come from ``mmcv-full``.
"""

import math

import torch


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


def _obb2poly_radian(boxes: torch.Tensor, version: str) -> torch.Tensor:
    """Match mmrotate's OBB-to-polygon conversion for radian boxes."""
    if boxes.numel() == 0:
        return boxes.new_zeros((boxes.shape[0], 8))
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f"Expected OBB tensor with shape [N, 5], got {tuple(boxes.shape)}")

    if version == "oc":
        x, y, width, height, angle = boxes.unbind(dim=1)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        width_x, width_y = width * 0.5 * cos_angle, width * 0.5 * sin_angle
        height_x, height_y = -height * 0.5 * sin_angle, height * 0.5 * cos_angle
        return torch.stack(
            [
                x - width_x - height_x,
                y - width_y - height_y,
                x + width_x - height_x,
                y + width_y - height_y,
                x + width_x + height_x,
                y + width_y + height_y,
                x - width_x + height_x,
                y - width_y + height_y,
            ],
            dim=-1,
        )

    if version not in {"le90", "le135"}:
        raise NotImplementedError(f"Unsupported angle version: {version}")

    center = boxes[:, :2]
    width, height, angle = boxes[:, 2], boxes[:, 3], boxes[:, 4]
    corners = torch.stack(
        [
            -width * 0.5,
            width * 0.5,
            width * 0.5,
            -width * 0.5,
            -height * 0.5,
            -height * 0.5,
            height * 0.5,
            height * 0.5,
        ],
        dim=0,
    ).reshape(2, 4, boxes.shape[0]).permute(2, 0, 1)
    sin_angle, cos_angle = torch.sin(angle), torch.cos(angle)
    rotation = torch.stack(
        [cos_angle, -sin_angle, sin_angle, cos_angle], dim=0
    ).reshape(2, 2, boxes.shape[0]).permute(2, 0, 1)
    polygons = rotation.matmul(corners).permute(2, 1, 0).reshape(-1, boxes.shape[0]).transpose(1, 0)
    polygons[:, 0::2] += center[:, 0].unsqueeze(1)
    polygons[:, 1::2] += center[:, 1].unsqueeze(1)
    return polygons.contiguous()


def _poly2obb_radian(polygons: torch.Tensor, version: str) -> torch.Tensor:
    """Match mmrotate's polygon-to-OBB conversion for radian boxes."""
    if polygons.numel() == 0:
        return polygons.new_zeros((polygons.shape[0], 5))
    if polygons.ndim < 2 or polygons.shape[-1] != 8:
        raise ValueError(f"Expected polygon tensor with final dimension 8, got {tuple(polygons.shape)}")

    # The source mmrotate helper flattens any intermediate class dimension,
    # e.g. [N, 1, 8] from the midpoint-offset RPN decoder becomes [N, 8].
    polygons = polygons.reshape(-1, 8)
    points = polygons.reshape(-1, 4, 2)
    if version == "oc":
        center = points.mean(dim=1)
        edge_1 = torch.linalg.vector_norm(points[:, 0] - points[:, 1], dim=1, keepdim=True)
        edge_2 = torch.linalg.vector_norm(points[:, 1] - points[:, 2], dim=1, keepdim=True)
        theta = torch.atan2(
            -(points[:, 1, 0] - points[:, 0, 0]),
            points[:, 1, 1] - points[:, 0, 1],
        ).unsqueeze(1)
        odd = torch.eq(torch.remainder(torch.floor(theta / (math.pi * 0.5)), 2), 0)
        width = torch.where(odd, edge_2, edge_1)
        height = torch.where(odd, edge_1, edge_2)
        angle = torch.remainder(theta, math.pi * 0.5)
        return torch.cat([center, width, height, angle], dim=1)

    if version not in {"le90", "le135"}:
        raise NotImplementedError(f"Unsupported angle version: {version}")

    point_1, point_2, point_3, point_4 = polygons.chunk(4, dim=1)
    edge_1 = torch.sqrt(
        torch.pow(point_1[:, 0] - point_2[:, 0], 2)
        + torch.pow(point_1[:, 1] - point_2[:, 1], 2)
    )
    edge_2 = torch.sqrt(
        torch.pow(point_2[:, 0] - point_3[:, 0], 2)
        + torch.pow(point_2[:, 1] - point_3[:, 1], 2)
    )
    angle_1 = torch.atan2(point_2[:, 1] - point_1[:, 1], point_2[:, 0] - point_1[:, 0])
    angle_2 = torch.atan2(point_4[:, 1] - point_1[:, 1], point_4[:, 0] - point_1[:, 0])
    angle = torch.where(edge_1 > edge_2, angle_1, angle_2)
    angle = norm_angle(angle, version, angle_unit=RADIAN)
    center_x = (point_1[:, 0] + point_3[:, 0]) * 0.5
    center_y = (point_1[:, 1] + point_3[:, 1]) * 0.5
    width = torch.maximum(edge_1, edge_2)
    height = torch.minimum(edge_1, edge_2)
    return torch.stack([center_x, center_y, width, height, angle], dim=1)


def obb2xyxy(boxes: torch.Tensor, version: str = "oc", angle_unit: str = DEGREE) -> torch.Tensor:
    radian_boxes = convert_obb_angle_unit(boxes, angle_unit, RADIAN)
    if radian_boxes.numel() == 0:
        return radian_boxes.new_zeros((radian_boxes.shape[0], 4))
    if radian_boxes.ndim != 2 or radian_boxes.shape[1] != 5:
        raise ValueError(f"Expected OBB tensor with shape [N, 5], got {tuple(radian_boxes.shape)}")

    if version == "oc":
        center = radian_boxes[:, :2]
        width, height, angle = radian_boxes[:, 2], radian_boxes[:, 3], radian_boxes[:, 4]
        hbb_width = torch.cos(angle) * width + torch.sin(angle) * height
        hbb_height = torch.sin(angle) * width + torch.cos(angle) * height
        half_size = torch.stack([hbb_width, hbb_height], dim=1) * 0.5
        return torch.cat([center - half_size, center + half_size], dim=1)
    if version == "le90":
        center = radian_boxes[:, :2]
        width, height, angle = radian_boxes[:, 2:3], radian_boxes[:, 3:4], radian_boxes[:, 4:5]
        x_bias = torch.abs(width * 0.5 * torch.cos(angle)) + torch.abs(height * 0.5 * torch.sin(angle))
        y_bias = torch.abs(width * 0.5 * torch.sin(angle)) + torch.abs(height * 0.5 * torch.cos(angle))
        bias = torch.cat([x_bias, y_bias], dim=1)
        return torch.cat([center - bias, center + bias], dim=1)
    if version == "le135":
        polygons = _obb2poly_radian(radian_boxes, version)
        x, y = polygons[:, 0::2], polygons[:, 1::2]
        return torch.stack([x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values], dim=1)
    raise NotImplementedError(f"Unsupported angle version: {version}")


def obb2poly(boxes: torch.Tensor, version: str = "oc", angle_unit: str = DEGREE) -> torch.Tensor:
    return _obb2poly_radian(convert_obb_angle_unit(boxes, angle_unit, RADIAN), version)


def poly2obb(polys: torch.Tensor, version: str = "oc", angle_unit: str = DEGREE) -> torch.Tensor:
    boxes = _poly2obb_radian(polys, version)
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
