from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LabelCatalog:
    categories: Optional[Any] = None
    predicates: Optional[Any] = None

    @staticmethod
    def infer_count(mapping: Any, default: int = 1) -> int:
        if mapping is None:
            return default
        if isinstance(mapping, list):
            return len(mapping) + 1
        if isinstance(mapping, dict):
            try:
                return max(int(v) for v in mapping.values()) + 1
            except Exception:
                return len(mapping) + 1
        return default

    @classmethod
    def from_annotation(cls, raw: Dict[str, Any]):
        return cls(
            categories=raw.get("categories"),
            predicates=raw.get("predicates"),
        )
