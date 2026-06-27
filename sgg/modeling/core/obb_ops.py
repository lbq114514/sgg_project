"""Light wrapper around mmrotate OBB utility functions."""

import math

import torch

try:
    from mmrotate.core import obb2xyxy, obb2poly, poly2obb
except ImportError as e:
    raise ImportError(
        "Failed to import mmrotate core OBB functions. Please install mmrotate."
    ) from e


def norm_angle(angle, angle_range="oc"):
    """
    Normalize rotated-box angles.

    The project currently uses degree-valued OBBs end-to-end, so normalization
    here follows degree conventions:
        - ``oc``   : [0, 90)
        - ``le90`` : [-90, 90)
        - ``le135``: [-45, 135)
    """
    if angle_range == "oc":
        period = 90.0
        start = 0.0
    elif angle_range == "le90":
        period = 180.0
        start = -90.0
    elif angle_range == "le135":
        period = 180.0
        start = -45.0
    else:
        raise NotImplementedError(f"Unsupported angle_range: {angle_range}")

    if isinstance(angle, torch.Tensor):
        return torch.remainder(angle - start, period) + start
    return ((angle - start) % period) + start


__all__ = ["obb2xyxy", "obb2poly", "poly2obb", "norm_angle"]
