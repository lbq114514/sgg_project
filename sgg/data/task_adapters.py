from __future__ import annotations

from typing import Literal

from sgg.structures.boxes import BoxList


SGGTask = Literal["predcls", "sgcls", "sgdet"]


def adapt_target_to_task(target: BoxList, task: SGGTask) -> BoxList:
    if task not in ("predcls", "sgcls", "sgdet"):
        raise ValueError(f"Unsupported SGG task: {task}")

    adapted = target
    adapted.add_field("sgg_task", task)
    adapted.add_field("use_gt_boxes", task in ("predcls", "sgcls"))
    adapted.add_field("use_gt_object_labels", task == "predcls")
    return adapted
