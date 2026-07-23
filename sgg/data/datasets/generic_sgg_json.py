from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..sample import DatasetMetadata, SGGSample
from ..sources import GenericSGGJsonSource
from ..task_adapters import adapt_target_to_task
from .registry import DATASETS
from .base import BaseSGGDataset
from sgg.structures.boxes import BoxList


@DATASETS.register("generic_sgg_json")
class GenericSGGJsonDataset(BaseSGGDataset):
    """
    Generic SGG dataset backed by a JSON annotation file.

    Expected format:
      {
        "images": [{
          "id": int,
          "file_name": str,
          "width": int,
          "height": int,
          "objects": [{"bbox": [...], "label": int}],
          "relations": [{"subject_id": int, "object_id": int, "predicate": int}]
        }],
        "categories": list | dict,
        "predicates": list | dict
      }
    """

    def __init__(
        self,
        ann_file: str,
        image_root: str,
        box_mode: str = "hbb",
        transforms=None,
        num_classes: Optional[int] = None,
        num_predicates: Optional[int] = None,
        filter_empty_relations: bool = False,
        filter_empty_objects: bool = True,
        deduplicate_relation_edges: bool = True,
        image_size: Optional[List[int]] = None,
        task: str = "sgdet",
        tile_enabled: bool = False,
        tile_size: Optional[List[int]] = None,
        tile_stride: Optional[List[int]] = None,
        tile_context: Optional[List[int]] = None,
        tile_min_objects: int = 1,
    ):
        self.source = GenericSGGJsonSource.from_paths(ann_file=ann_file, image_root=image_root)
        metadata = DatasetMetadata(
            dataset_name="generic_sgg_json",
            box_mode=box_mode,
            num_classes=num_classes or self.source.num_classes,
            num_predicates=num_predicates or self.source.num_predicates,
            categories=self.source.categories,
            predicates=self.source.predicates,
        )
        super().__init__(transforms=transforms, metadata=metadata)

        self.ann_file = Path(ann_file)
        self.image_root = Path(image_root)
        self.box_mode = box_mode
        self.image_size = image_size
        self.task = task
        self.tile_enabled = bool(tile_enabled)
        self.tile_size = tuple(tile_size or (1024, 1024))
        self.tile_stride = tuple(tile_stride or self.tile_size)
        self.tile_context = tuple(tile_context or (0, 0))
        self.tile_min_objects = int(tile_min_objects)

        self.records = self.source.get_records(
            filter_empty_objects=filter_empty_objects,
            filter_empty_relations=filter_empty_relations,
            deduplicate_relation_edges=deduplicate_relation_edges,
        )
        if self.tile_enabled:
            self.records = self._tile_records(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def _tile_starts(self, length: int, window: int, stride: int) -> List[int]:
        if length <= window:
            return [0]
        starts = []
        pos = 0
        while True:
            starts.append(pos)
            if pos + window >= length:
                break
            pos = min(pos + stride, max(length - window, 0))
            if pos == starts[-1]:
                break
        return starts

    def _tile_record(self, record: Dict[str, Any], x0: int, y0: int, tile_w: int, tile_h: int):
        x1 = min(x0 + tile_w, int(record["width"]))
        y1 = min(y0 + tile_h, int(record["height"]))
        ctx_h, ctx_w = int(self.tile_context[0]), int(self.tile_context[1])
        crop_x0 = max(x0 - ctx_w, 0)
        crop_y0 = max(y0 - ctx_h, 0)
        crop_x1 = min(x1 + ctx_w, int(record["width"]))
        crop_y1 = min(y1 + ctx_h, int(record["height"]))
        width = crop_x1 - crop_x0
        height = crop_y1 - crop_y0
        objects = []
        idx_map = {}
        for obj_idx, obj in enumerate(record.get("objects", [])):
            bbox = obj["bbox"]
            if self.box_mode == "obb":
                cx, cy, w, h, a = bbox
                keep = (crop_x0 <= cx < crop_x1) and (crop_y0 <= cy < crop_y1)
                if not keep:
                    continue
                new_bbox = [cx - crop_x0, cy - crop_y0, w, h, a]
            else:
                bx1, by1, bx2, by2 = bbox
                cx = 0.5 * (bx1 + bx2)
                cy = 0.5 * (by1 + by2)
                keep = (crop_x0 <= cx < crop_x1) and (crop_y0 <= cy < crop_y1)
                if not keep:
                    continue
                nx1 = max(float(bx1) - crop_x0, 0.0)
                ny1 = max(float(by1) - crop_y0, 0.0)
                nx2 = min(float(bx2) - crop_x0, float(width))
                ny2 = min(float(by2) - crop_y0, float(height))
                if nx2 <= nx1 or ny2 <= ny1:
                    continue
                new_bbox = [nx1, ny1, nx2, ny2]
            idx_map[obj_idx] = len(objects)
            objects.append({"bbox": new_bbox, "label": int(obj["label"])})

        if len(objects) < self.tile_min_objects:
            return None

        relations = []
        for rel in record.get("relations", []):
            subj = int(rel["subject_id"])
            obj = int(rel["object_id"])
            if subj in idx_map and obj in idx_map and idx_map[subj] != idx_map[obj]:
                relations.append(
                    {
                        "subject_id": idx_map[subj],
                        "object_id": idx_map[obj],
                        "predicate": int(rel["predicate"]),
                    }
                )

        tiled = dict(record)
        tiled["width"] = width
        tiled["height"] = height
        tiled["objects"] = objects
        tiled["relations"] = relations
        tiled["_tile"] = {"x0": crop_x0, "y0": crop_y0, "x1": crop_x1, "y1": crop_y1}
        tiled["_tile_core"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        tiled["_source_image_id"] = int(record.get("id", 0))
        tiled["_source_width"] = int(record["width"])
        tiled["_source_height"] = int(record["height"])
        return tiled

    def _tile_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tile_h, tile_w = int(self.tile_size[0]), int(self.tile_size[1])
        stride_h, stride_w = int(self.tile_stride[0]), int(self.tile_stride[1])
        tiled_records = []
        for record in records:
            width = int(record["width"])
            height = int(record["height"])
            x_starts = self._tile_starts(width, tile_w, stride_w)
            y_starts = self._tile_starts(height, tile_h, stride_h)
            if len(x_starts) == 1 and len(y_starts) == 1 and x_starts[0] == 0 and y_starts[0] == 0:
                tiled_records.append(record)
                continue
            for y0 in y_starts:
                for x0 in x_starts:
                    tiled = self._tile_record(record, x0, y0, tile_w, tile_h)
                    if tiled is not None:
                        tiled_records.append(tiled)
        return tiled_records

    def _resize_image_if_needed(self, image: torch.Tensor):
        if self.image_size is None:
            _, height, width = image.shape
            return image, width, height, 1.0, 1.0

        _, in_height, in_width = image.shape
        out_height, out_width = self.image_size

        image = F.interpolate(
            image.unsqueeze(0),
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        scale_x = out_width / max(in_width, 1)
        scale_y = out_height / max(in_height, 1)
        return image, out_width, out_height, scale_x, scale_y

    def _build_relation_fields(self, n_obj: int, relations: List[Dict[str, Any]]):
        pair_labels = torch.zeros((n_obj, n_obj), dtype=torch.long)
        rel_triplets = []
        for rel in relations:
            s = int(rel["subject_id"])
            o = int(rel["object_id"])
            p = int(rel["predicate"])
            if 0 <= s < n_obj and 0 <= o < n_obj and s != o:
                pair_labels[s, o] = p
                rel_triplets.append([s, o, p])

        if rel_triplets:
            rel_triplets_t = torch.tensor(rel_triplets, dtype=torch.long)
        else:
            rel_triplets_t = torch.zeros((0, 3), dtype=torch.long)
        return pair_labels, rel_triplets_t

    def _build_target(
        self,
        record: Dict[str, Any],
        width: int,
        height: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        core_box: Optional[List[float]] = None,
    ) -> BoxList:
        objects = record.get("objects", [])
        box_dim = 4 if self.box_mode == "hbb" else 5
        if objects:
            boxes = torch.tensor([obj["bbox"] for obj in objects], dtype=torch.float32)
            if self.box_mode == "hbb":
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
            else:
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
            labels = torch.tensor([int(obj["label"]) for obj in objects], dtype=torch.long)
        else:
            boxes = torch.zeros((0, box_dim), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        pair_labels, rel_triplets = self._build_relation_fields(
            len(objects),
            record.get("relations", []),
        )

        mode = "xyxy" if self.box_mode == "hbb" else "xywha"
        target = BoxList(boxes, (width, height), mode=mode)
        target.add_field("labels", labels)
        target.add_field("pair_labels", pair_labels)
        target.add_field("relation_triplets", rel_triplets)
        target.add_field("image_id", torch.tensor([int(record.get("id", 0))], dtype=torch.long))
        if core_box is not None:
            x0, y0, x1, y1 = core_box
            if self.box_mode == "obb":
                centers = boxes[:, :2] if boxes.numel() > 0 else boxes.new_zeros((0, 2))
            else:
                centers = 0.5 * (boxes[:, :2] + boxes[:, 2:]) if boxes.numel() > 0 else boxes.new_zeros((0, 2))
            is_core = (
                (centers[:, 0] >= x0)
                & (centers[:, 0] < x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] < y1)
            )
            target.add_field("is_core", is_core.to(dtype=torch.bool))
            target.add_field("tile_core_box", torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32))
        return adapt_target_to_task(target, self.task)

    def get_item_dict(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = self.source.load_image(record["file_name"])
        tile = record.get("_tile")
        core_box = None
        if tile is not None:
            image = image[:, tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]]
        image, width, height, scale_x, scale_y = self._resize_image_if_needed(image)
        if tile is not None:
            core = record.get("_tile_core", tile)
            core_box = [
                (float(core["x0"]) - float(tile["x0"])) * scale_x,
                (float(core["y0"]) - float(tile["y0"])) * scale_y,
                (float(core["x1"]) - float(tile["x0"])) * scale_x,
                (float(core["y1"]) - float(tile["y0"])) * scale_y,
            ]
        target = self._build_target(
            record,
            width=width,
            height=height,
            scale_x=scale_x,
            scale_y=scale_y,
            core_box=core_box,
        )
        meta = {
            "image_id": int(record.get("id", index)),
            "file_name": record["file_name"],
            "height": height,
            "width": width,
            "dataset_name": self.metadata.dataset_name,
        }
        if tile is not None:
            meta["tile_origin"] = (int(tile["x0"]), int(tile["y0"]))
            meta["tile_scale"] = (float(scale_x), float(scale_y))
            meta["source_image_id"] = int(record.get("_source_image_id", record.get("id", index)))
            meta["source_width"] = int(record.get("_source_width", record.get("width", width)))
            meta["source_height"] = int(record.get("_source_height", record.get("height", height)))
            core = record.get("_tile_core", tile)
            meta["tile_core"] = (
                int(core["x0"]),
                int(core["y0"]),
                int(core["x1"]),
                int(core["y1"]),
            )
        return SGGSample(image=image, target=target, meta=meta)
