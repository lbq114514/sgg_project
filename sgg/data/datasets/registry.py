from __future__ import annotations

from typing import Callable, Dict, Type


class DatasetRegistry:
    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable:
        def decorator(cls):
            if name in self._registry:
                raise KeyError(f"Dataset '{name}' is already registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"Dataset '{name}' is not registered")
        return self._registry[name]

    def keys(self):
        return list(self._registry.keys())

    def as_dict(self) -> Dict[str, Type]:
        return dict(self._registry)


DATASETS = DatasetRegistry()
