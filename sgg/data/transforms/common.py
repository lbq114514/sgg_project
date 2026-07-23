from __future__ import annotations

import math
import random
from typing import Tuple

import torch

from sgg.modeling.core.obb_ops import get_boxlist_angle_unit, norm_angle, obb2poly, poly2obb
from sgg.structures.boxes import BoxList


class IdentityTransform:
    def __call__(self, image: torch.Tensor, target: BoxList):
        return image, target


class ResizeTransform:
    def __init__(self, size: Tuple[int, int], keep_ratio: bool = False):
        self.size = size
        self.keep_ratio = bool(keep_ratio)

    def __call__(self, image: torch.Tensor, target: BoxList):
        _, in_h, in_w = image.shape
        target_h, target_w = self.size
        if self.keep_ratio:
            scale = min(target_w / max(in_w, 1), target_h / max(in_h, 1))
            out_w = max(int(round(in_w * scale)), 1)
            out_h = max(int(round(in_h * scale)), 1)
        else:
            out_h, out_w = target_h, target_w
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        boxes = target.bbox.clone()
        scale_x = out_w / max(in_w, 1)
        scale_y = out_h / max(in_h, 1)
        if target.mode == "xyxy":
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        elif target.mode == "xywha":
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
        else:
            raise ValueError(f"Unsupported target mode: {target.mode}")

        resized = BoxList(boxes, (out_w, out_h), mode=target.mode)
        for field in target.fields():
            value = target.get_field(field)
            if field == "tile_core_box":
                value = value.clone()
                value[:, [0, 2]] *= scale_x
                value[:, [1, 3]] *= scale_y
            resized.add_field(field, value)
        return image, resized


class NormalizeTransform:
    def __init__(self, mean, std, to_rgb: bool = True):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.max() > 1.0 or std.max() > 1.0:
            mean = mean / 255.0
            std = std / 255.0
        self.mean = mean.view(3, 1, 1)
        self.std = std.view(3, 1, 1)
        self.to_rgb = bool(to_rgb)

    def __call__(self, image: torch.Tensor, target: BoxList):
        # Images are loaded through PIL as RGB tensors in [0, 1]. In this
        # pipeline `to_rgb=True` means the tensor is already in the target RGB
        # order, matching MMDetection's normalized channel convention.
        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: BoxList):
        if random.random() >= self.prob:
            return image, target

        image = torch.flip(image, dims=[2])
        width = target.size[0]
        boxes = target.bbox.clone()
        if target.mode == "xyxy":
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes[:, 0] = x1
            boxes[:, 2] = x2
        elif target.mode == "xywha":
            boxes[:, 0] = width - boxes[:, 0]
            boxes[:, 4] = -boxes[:, 4]
        else:
            raise ValueError(f"Unsupported target mode: {target.mode}")

        flipped = BoxList(boxes, target.size, mode=target.mode)
        for field in target.fields():
            flipped.add_field(field, target.get_field(field))
        return image, flipped


class RandomDirectionalFlip:
    def __init__(self, probs=(0.25, 0.25, 0.25), directions=("horizontal", "vertical", "diagonal"), angle_version: str = "le90"):
        self.probs = tuple(float(p) for p in probs)
        self.directions = tuple(directions)
        self.angle_version = angle_version

    def __call__(self, image: torch.Tensor, target: BoxList):
        total_prob = sum(self.probs)
        if total_prob <= 0:
            return image, target
        draw = random.random()
        if draw >= total_prob:
            return image, target

        acc = 0.0
        direction = self.directions[0]
        for p, d in zip(self.probs, self.directions):
            acc += p
            if draw < acc:
                direction = d
                break

        return _flip_with_direction(image, target, direction, self.angle_version)


def _flip_with_direction(image: torch.Tensor, target: BoxList, direction: str, angle_version: str):
    if direction == "horizontal":
        image = torch.flip(image, dims=[2])
    elif direction == "vertical":
        image = torch.flip(image, dims=[1])
    elif direction == "diagonal":
        image = torch.flip(image, dims=[1, 2])
    else:
        raise ValueError(f"Unsupported flip direction: {direction}")

    width, height = target.size
    boxes = target.bbox.clone()
    angle_unit = get_boxlist_angle_unit(target)
    if target.mode == "xyxy":
        if direction in {"horizontal", "diagonal"}:
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = x1, x2
        if direction in {"vertical", "diagonal"}:
            y1 = height - boxes[:, 3]
            y2 = height - boxes[:, 1]
            boxes[:, 1], boxes[:, 3] = y1, y2
    elif target.mode == "xywha":
        if direction in {"horizontal", "diagonal"}:
            boxes[:, 0] = width - boxes[:, 0]
        if direction in {"vertical", "diagonal"}:
            boxes[:, 1] = height - boxes[:, 1]
        if direction == "horizontal":
            boxes[:, 4] = -boxes[:, 4]
        elif direction == "vertical":
            boxes[:, 4] = -boxes[:, 4]
        else:
            boxes[:, 4] = boxes[:, 4] + (math.pi if angle_unit == "radian" else 180.0)
        boxes[:, 4] = norm_angle(boxes[:, 4], angle_version, angle_unit=angle_unit)
    else:
        raise ValueError(f"Unsupported target mode: {target.mode}")

    flipped = BoxList(boxes, target.size, mode=target.mode)
    for field in target.fields():
        value = target.get_field(field)
        if torch.is_tensor(value):
            value = value.clone()
        flipped.add_field(field, value)
    return image, flipped


class RandomOBBRotate:
    def __init__(self, rotate_ratio: float = 0.5, angles_range: float = 180.0, angle_version: str = "le90"):
        self.rotate_ratio = float(rotate_ratio)
        self.angles_range = float(angles_range)
        self.angle_version = angle_version

    def __call__(self, image: torch.Tensor, target: BoxList):
        if target.mode != "xywha" or random.random() >= self.rotate_ratio:
            return image, target

        angle = random.uniform(-0.5 * self.angles_range, 0.5 * self.angles_range)
        _, height, width = image.shape
        center = image.new_tensor([width * 0.5, height * 0.5], dtype=torch.float32)
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        theta = image.new_tensor(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
            ],
            dtype=image.dtype,
        )
        grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        rotated_image = torch.nn.functional.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        angle_unit = get_boxlist_angle_unit(target)
        polys = obb2poly(target.bbox, self.angle_version, angle_unit=angle_unit).view(-1, 4, 2)
        centered = polys - center[None, None, :]
        rot = image.new_tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        rotated_polys = centered @ rot.T + center[None, None, :]
        rotated_boxes = poly2obb(rotated_polys.reshape(-1, 8), self.angle_version, angle_unit=angle_unit)
        rotated_boxes[:, 4] = norm_angle(rotated_boxes[:, 4], self.angle_version, angle_unit=angle_unit)

        rotated_target = BoxList(rotated_boxes.to(dtype=target.bbox.dtype), target.size, target.mode)
        for field in target.fields():
            value = target.get_field(field)
            if torch.is_tensor(value):
                value = value.clone()
            rotated_target.add_field(field, value)
        return rotated_image, rotated_target
