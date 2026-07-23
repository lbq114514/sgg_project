from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_tensor(
    path: str | Path,
    *,
    allow_large_images: bool = False,
) -> torch.Tensor:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError("Please install pillow and numpy to load images.") from exc

    # Pillow protects generic applications from unexpectedly large compressed
    # images.  Trusted remote-sensing datasets routinely exceed that limit,
    # so callers must opt in explicitly instead of disabling the protection
    # process-wide for every image source.
    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    if allow_large_images:
        Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(Path(path)) as img:
            arr = np.asarray(img.convert("RGB")).copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
