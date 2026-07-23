import ast
from pathlib import Path

import pytest
import torch

from sgg.modeling.core.obb_ops import RADIAN, obb2poly, obb2xyxy, poly2obb


@pytest.mark.parametrize("version", ["oc", "le90", "le135"])
def test_local_obb_helpers_round_trip(version):
    boxes = torch.tensor(
        [[10.0, 20.0, 8.0, 4.0, 0.3], [1.0, 2.0, 7.0, 3.0, 1.1707963]],
        dtype=torch.float32,
    )
    polygons = obb2poly(boxes, version=version, angle_unit=RADIAN)
    horizontal = obb2xyxy(boxes, version=version, angle_unit=RADIAN)
    restored = poly2obb(polygons, version=version, angle_unit=RADIAN)

    assert polygons.shape == (2, 8)
    assert horizontal.shape == (2, 4)
    assert torch.allclose(restored, boxes, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("version", ["oc", "le90", "le135"])
def test_local_obb_helpers_handle_empty_tensors(version):
    assert obb2poly(torch.empty(0, 5), version=version).shape == (0, 8)
    assert obb2xyxy(torch.empty(0, 5), version=version).shape == (0, 4)
    assert poly2obb(torch.empty(0, 8), version=version).shape == (0, 5)


def test_obb_ops_has_no_mmrotate_import():
    source = Path("sgg/modeling/core/obb_ops.py").read_text()
    imported_roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "mmrotate" not in imported_roots
    assert "mmdet" not in imported_roots

