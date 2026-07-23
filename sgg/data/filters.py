from __future__ import annotations

from typing import Any, Dict, Iterable, List


def filter_empty_relations(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if len(record.get("relations", [])) > 0]


def filter_empty_objects(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if len(record.get("objects", [])) > 0]


def deduplicate_relations(relations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()
    for rel in relations:
        key = (int(rel["subject_id"]), int(rel["object_id"]), int(rel["predicate"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped
