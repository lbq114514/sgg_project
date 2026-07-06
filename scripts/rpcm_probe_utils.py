from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


def _as_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.detach().float().cpu()
    return tensor.detach().cpu()


def tensor_digest(
    tensor: torch.Tensor,
    *,
    max_rows: int = 64,
    max_cols: int = 64,
    max_flat: int = 4096,
) -> dict[str, Any]:
    value = tensor.detach()
    cpu = _as_float_tensor(value)
    flat = cpu.reshape(-1)
    finite = torch.isfinite(cpu.float()).all().item() if cpu.numel() else True
    stats_tensor = cpu.float()
    digest: dict[str, Any] = {
        "shape": tuple(cpu.shape),
        "dtype": str(value.dtype),
        "numel": int(cpu.numel()),
        "finite": bool(finite),
    }
    if cpu.numel():
        digest.update(
            {
                "mean": float(stats_tensor.mean().item()),
                "std": float(stats_tensor.std(unbiased=False).item()),
                "min": float(stats_tensor.min().item()),
                "max": float(stats_tensor.max().item()),
                "norm": float(stats_tensor.norm().item()),
            }
        )
    else:
        digest.update({"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0})

    if cpu.dim() == 0:
        sample = cpu.reshape(1)
    elif cpu.dim() == 1:
        sample = cpu[:max_flat]
    elif cpu.dim() == 2:
        sample = cpu[:max_rows, :max_cols]
    else:
        sample = flat[:max_flat]
    digest["sample"] = sample.contiguous()
    return digest


def summarize_value(
    value: Any,
    *,
    max_rows: int = 64,
    max_cols: int = 64,
    max_flat: int = 4096,
) -> Any:
    if torch.is_tensor(value):
        return tensor_digest(value, max_rows=max_rows, max_cols=max_cols, max_flat=max_flat)
    if isinstance(value, dict):
        return OrderedDict(
            (str(k), summarize_value(v, max_rows=max_rows, max_cols=max_cols, max_flat=max_flat))
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return [
            summarize_value(v, max_rows=max_rows, max_cols=max_cols, max_flat=max_flat)
            for v in value
        ]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(type(value))


class HookDumper:
    def __init__(self, *, max_rows: int = 64, max_cols: int = 64, max_flat: int = 4096):
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)
        self.max_flat = int(max_flat)
        self.records: OrderedDict[str, Any] = OrderedDict()
        self._handles = []
        self._counts: dict[str, int] = {}

    def _store(self, name: str, output: Any) -> None:
        count = self._counts.get(name, 0)
        self._counts[name] = count + 1
        key = f"{name}#{count}"
        self.records[key] = summarize_value(
            output,
            max_rows=self.max_rows,
            max_cols=self.max_cols,
            max_flat=self.max_flat,
        )

    def add(self, module: torch.nn.Module | None, name: str) -> None:
        if module is None:
            return
        self._handles.append(module.register_forward_hook(lambda _m, _i, o, n=name: self._store(n, o)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def compare_samples(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "left_shape": left.get("shape"),
        "right_shape": right.get("shape"),
        "left_mean": left.get("mean"),
        "right_mean": right.get("mean"),
        "left_std": left.get("std"),
        "right_std": right.get("std"),
        "left_norm": left.get("norm"),
        "right_norm": right.get("norm"),
    }
    ls = left.get("sample")
    rs = right.get("sample")
    if torch.is_tensor(ls) and torch.is_tensor(rs) and tuple(ls.shape) == tuple(rs.shape):
        lf = ls.float().reshape(-1)
        rf = rs.float().reshape(-1)
        diff = (lf - rf).abs()
        out.update(
            {
                "sample_max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "sample_mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                "sample_cosine": float(torch.nn.functional.cosine_similarity(lf, rf, dim=0).item())
                if lf.numel() and rf.numel()
                else 1.0,
            }
        )
    else:
        out["sample_compare"] = "missing_or_shape_mismatch"
    return out
