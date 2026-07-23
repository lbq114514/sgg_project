from __future__ import annotations

from typing import Dict, List

import torch


def summarize_dataset(dataset) -> Dict[str, float]:
    num_images = len(dataset)
    num_objects = 0
    num_relations = 0
    predicate_hist = torch.zeros(dataset.metadata.num_predicates, dtype=torch.long)

    for record in dataset.records:
        num_objects += len(record.get("objects", []))
        rels = record.get("relations", [])
        num_relations += len(rels)
        for rel in rels:
            pred = int(rel["predicate"])
            if 0 <= pred < predicate_hist.numel():
                predicate_hist[pred] += 1

    return {
        "num_images": float(num_images),
        "num_objects": float(num_objects),
        "num_relations": float(num_relations),
        "avg_objects_per_image": float(num_objects / max(num_images, 1)),
        "avg_relations_per_image": float(num_relations / max(num_images, 1)),
        "num_nonzero_predicates": float((predicate_hist > 0).sum().item()),
    }


def predicate_histogram(dataset) -> List[int]:
    predicate_hist = torch.zeros(dataset.metadata.num_predicates, dtype=torch.long)
    for record in dataset.records:
        for rel in record.get("relations", []):
            pred = int(rel["predicate"] if isinstance(rel, dict) else rel[2])
            if 0 <= pred < predicate_hist.numel():
                predicate_hist[pred] += 1
    return [int(v) for v in predicate_hist.tolist()]
