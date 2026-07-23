from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..catalog import LabelCatalog
from ..filters import (
    deduplicate_relations,
    filter_empty_objects as apply_filter_empty_objects,
    filter_empty_relations as apply_filter_empty_relations,
)
from ..io import load_image_tensor, load_json


class GenericSGGJsonSource:
    _SOURCE_CACHE: ClassVar[Dict[Tuple[str, str], "GenericSGGJsonSource"]] = {}
    _RECORDS_CACHE: ClassVar[Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = {}

    def __init__(self, ann_file: str, image_root: str):
        self.ann_file = Path(ann_file)
        self.image_root = Path(image_root)
        self.raw = load_json(self.ann_file)
        self.catalog = LabelCatalog.from_annotation(self.raw)

    @classmethod
    def from_paths(cls, ann_file: str, image_root: str) -> "GenericSGGJsonSource":
        cache_key = (str(Path(ann_file).resolve()), str(Path(image_root).resolve()))
        source = cls._SOURCE_CACHE.get(cache_key)
        if source is None:
            source = cls(ann_file=ann_file, image_root=image_root)
            cls._SOURCE_CACHE[cache_key] = source
        return source

    @classmethod
    def clear_cache(cls) -> None:
        cls._SOURCE_CACHE.clear()
        cls._RECORDS_CACHE.clear()

    @property
    def categories(self):
        return self.catalog.categories

    @property
    def predicates(self):
        return self.catalog.predicates

    @property
    def num_classes(self) -> int:
        return self.catalog.infer_count(self.catalog.categories, default=1)

    @property
    def num_predicates(self) -> int:
        return self.catalog.infer_count(self.catalog.predicates, default=1)

    def get_records(
        self,
        filter_empty_objects: bool = True,
        filter_empty_relations: bool = False,
        deduplicate_relation_edges: bool = True,
    ) -> List[Dict[str, Any]]:
        cache_key = (
            str(self.ann_file.resolve()),
            str(self.image_root.resolve()),
            filter_empty_objects,
            filter_empty_relations,
            deduplicate_relation_edges,
        )
        records = self._RECORDS_CACHE.get(cache_key)
        if records is None:
            records = list(self.raw["images"])
            if filter_empty_objects:
                records = apply_filter_empty_objects(records)
            if filter_empty_relations:
                records = apply_filter_empty_relations(records)
            if deduplicate_relation_edges:
                deduped = []
                for record in records:
                    new_record = dict(record)
                    new_record["relations"] = deduplicate_relations(record.get("relations", []))
                    deduped.append(new_record)
                records = deduped
            self._RECORDS_CACHE[cache_key] = records
        return records

    def load_image(self, file_name: str):
        return load_image_tensor(self.image_root / file_name)
