from pathlib import Path

import pytest

from sgg.data.io import load_image_tensor


def _write_test_image(path: Path) -> None:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)


def test_large_image_opt_in_bypasses_pillow_limit_and_restores_it(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "large-for-test.png"
    _write_test_image(image_path)

    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1
    try:
        tensor = load_image_tensor(image_path, allow_large_images=True)
        assert tensor.shape == (3, 4, 4)
        assert Image.MAX_IMAGE_PIXELS == 1
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels


def test_generic_image_loader_keeps_pillow_protection_enabled(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "large-for-test.png"
    _write_test_image(image_path)

    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1
    try:
        with pytest.raises(Image.DecompressionBombError):
            load_image_tensor(image_path)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
