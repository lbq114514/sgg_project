from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ..sample import DatasetMetadata, SGGSample
from ..sources import STARSource
from ..task_adapters import adapt_target_to_task
from .base import BaseSGGDataset
from .registry import DATASETS
from sgg.structures.boxes import BoxList
from sgg.modeling.core.obb_ops import normalize_angle_unit, set_boxlist_angle_unit


@DATASETS.register("star")
class STARDataset(BaseSGGDataset):
    def __init__(
        self,
        split: str,
        image_root: Optional[str] = None,
        roidb_file: Optional[str] = None,
        dict_file: Optional[str] = None,
        image_file: Optional[str] = None,
        img_dir: Optional[str] = None,
        box_mode: str = "hbb",
        transforms=None,
        task: str = "sgdet",
        num_classes: Optional[int] = None,
        num_predicates: Optional[int] = None,
        filter_empty_relations: bool = True,
        filter_duplicate_relations: bool = True,
        filter_non_overlap: bool = False,
        num_im: int = -1,
        split_mode: str = "fixed",
        split_ratios: Sequence[int] = (6, 2, 2),
        random_seed: int = 42,
        fixed_split_source: Optional[str] = None,
        image_ext: str = ".png",
        tile_enabled: bool = False,
        tile_size: Optional[Sequence[int]] = None,
        tile_stride: Optional[Sequence[int]] = None,
        tile_context: Optional[Sequence[int]] = None,
        tile_min_objects: int = 1,
        box_angle_unit: str = "degree",
        box_coord_scale: Optional[float] = None,
        sgdet_detector_original_resolution: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if box_mode not in {"hbb", "obb"}:
            raise ValueError(f"Unsupported box_mode: {box_mode}")

        image_root = image_root or img_dir
        if not image_root:
            raise ValueError("STARDataset requires `image_root` (or `img_dir`).")
        if not roidb_file or not dict_file or not image_file:
            raise ValueError("STARDataset requires `roidb_file`, `dict_file`, and `image_file`.")

        self.split = split
        self.image_root = Path(image_root)
        self.box_mode = box_mode
        self.box_angle_unit = normalize_angle_unit(box_angle_unit)
        self.task = task
        self.filter_duplicate_relations = filter_duplicate_relations and split == "train"
        self.tile_enabled = bool(tile_enabled)
        self.tile_size = tuple(tile_size or (1024, 1024))
        self.tile_stride = tuple(tile_stride or self.tile_size)
        self.tile_context = tuple(tile_context or (0, 0))
        self.tile_min_objects = int(tile_min_objects)

        self.source = STARSource.from_paths(
            image_root=self.image_root,
            roidb_file=Path(roidb_file),
            dict_file=Path(dict_file),
            image_file=Path(image_file),
            image_ext=image_ext,
        )

        metadata = DatasetMetadata(
            dataset_name="star",
            box_mode=box_mode,
            num_classes=num_classes or len(self.source.ind_to_classes),
            num_predicates=num_predicates or len(self.source.ind_to_predicates),
            categories={i: name for i, name in enumerate(self.source.ind_to_classes)},
            predicates={i: name for i, name in enumerate(self.source.ind_to_predicates)},
        )
        super().__init__(
            transforms=transforms,
            metadata=metadata,
            sgdet_detector_original_resolution=sgdet_detector_original_resolution,
        )

        self.ind_to_classes = self.source.ind_to_classes
        self.ind_to_predicates = self.source.ind_to_predicates
        self.ind_to_attributes = self.source.ind_to_attributes
        self.records = self.source.get_split_records(
            split=split,
            box_mode=box_mode,
            split_mode=split_mode,
            split_ratios=split_ratios,
            random_seed=random_seed,
            filter_empty_relations=filter_empty_relations,
            filter_non_overlap=filter_non_overlap and split == "train",
            num_im=num_im,
            box_coord_scale=box_coord_scale,
        )
        if self.tile_enabled:
            self.records = self._tile_records(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def _tile_starts(self, length: int, window: int, stride: int):
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
        boxes = []
        labels = []
        attrs = []
        idx_map: Dict[int, int] = {}

        for obj_idx, (box, label) in enumerate(zip(record["boxes"], record["labels"])):
            if self.box_mode == "obb":
                cx, cy, w, h, a = [float(v) for v in box.tolist()]
                keep = (crop_x0 <= cx < crop_x1) and (crop_y0 <= cy < crop_y1)
                if not keep:
                    continue
                new_box = [cx - crop_x0, cy - crop_y0, w, h, a]
            else:
                bx1, by1, bx2, by2 = [float(v) for v in box.tolist()]
                cx = 0.5 * (bx1 + bx2)
                cy = 0.5 * (by1 + by2)
                keep = (crop_x0 <= cx < crop_x1) and (crop_y0 <= cy < crop_y1)
                if not keep:
                    continue
                nx1 = max(bx1 - crop_x0, 0.0)
                ny1 = max(by1 - crop_y0, 0.0)
                nx2 = min(bx2 - crop_x0, float(width))
                ny2 = min(by2 - crop_y0, float(height))
                if nx2 <= nx1 or ny2 <= ny1:
                    continue
                new_box = [nx1, ny1, nx2, ny2]
            idx_map[obj_idx] = len(boxes)
            boxes.append(new_box)
            labels.append(int(label))
            if "attributes" in record and len(record["attributes"]) > obj_idx:
                attrs.append(record["attributes"][obj_idx])

        if len(boxes) < self.tile_min_objects:
            return None

        relations = []
        for rel in record["relations"]:
            subj, obj, pred = [int(v) for v in rel]
            if subj in idx_map and obj in idx_map and idx_map[subj] != idx_map[obj]:
                relations.append([idx_map[subj], idx_map[obj], pred])

        tiled = dict(record)
        tiled["width"] = width
        tiled["height"] = height
        tiled["boxes"] = np.asarray(boxes, dtype=np.float32)
        tiled["labels"] = np.asarray(labels, dtype=np.int64)
        tiled["relations"] = np.asarray(relations, dtype=np.int64) if relations else np.zeros((0, 3), dtype=np.int64)
        if attrs:
            tiled["attributes"] = np.asarray(attrs)
        tiled["_tile"] = {"x0": crop_x0, "y0": crop_y0, "x1": crop_x1, "y1": crop_y1}
        tiled["_tile_core"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        tiled["_source_image_id"] = int(record["image_index"])
        tiled["_source_width"] = int(record["width"])
        tiled["_source_height"] = int(record["height"])
        return tiled

    def _tile_records(self, records):
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

    @classmethod
    def clear_cache(cls) -> None:
        STARSource.clear_cache()

    def _build_relation_fields(self, relations, num_objects: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_tensor = torch.as_tensor(relations, dtype=torch.long)
        if rel_tensor.numel() == 0:
            empty = torch.zeros((0, 3), dtype=torch.long)
            return torch.zeros((num_objects, num_objects), dtype=torch.long), empty, empty

        all_kept = []
        for subj, obj, pred in rel_tensor.tolist():
            if 0 <= subj < num_objects and 0 <= obj < num_objects and subj != obj:
                all_kept.append([subj, obj, pred])
        all_rel_triplets = (
            torch.tensor(all_kept, dtype=torch.long)
            if all_kept else torch.zeros((0, 3), dtype=torch.long)
        )

        if self.filter_duplicate_relations:
            grouped: Dict[Tuple[int, int], list[int]] = {}
            for subj, obj, pred in rel_tensor.tolist():
                grouped.setdefault((int(subj), int(obj)), []).append(int(pred))
            sampled = [
                [subj, obj, random.choice(preds)]
                for (subj, obj), preds in grouped.items()
            ]
            rel_tensor = torch.tensor(sampled, dtype=torch.long)

        pair_labels = torch.zeros((num_objects, num_objects), dtype=torch.long)
        kept = []
        for subj, obj, pred in rel_tensor.tolist():
            if 0 <= subj < num_objects and 0 <= obj < num_objects and subj != obj:
                pair_labels[subj, obj] = pred
                kept.append([subj, obj, pred])

        rel_triplets = torch.tensor(kept, dtype=torch.long) if kept else torch.zeros((0, 3), dtype=torch.long)
        return pair_labels, rel_triplets, all_rel_triplets

    def get_item_dict(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = self.source.load_image(record["file_name"])
        tile = record.get("_tile")
        core_box = None
        if tile is not None:
            image = image[:, tile["y0"] : tile["y1"], tile["x0"] : tile["x1"]]
        boxes = torch.as_tensor(record["boxes"], dtype=torch.float32)
        if self.box_mode == "obb" and self.box_angle_unit == "radian" and boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 4] = torch.deg2rad(boxes[:, 4])
        labels = torch.as_tensor(record["labels"], dtype=torch.long)
        pair_labels, rel_triplets, all_rel_triplets = self._build_relation_fields(record["relations"], len(labels))
        if tile is not None:
            core = record.get("_tile_core", tile)
            core_box = [
                float(core["x0"]) - float(tile["x0"]),
                float(core["y0"]) - float(tile["y0"]),
                float(core["x1"]) - float(tile["x0"]),
                float(core["y1"]) - float(tile["y0"]),
            ]

        mode = "xyxy" if self.box_mode == "hbb" else "xywha"
        target = BoxList(boxes, (record["width"], record["height"]), mode=mode)
        if mode == "xywha":
            set_boxlist_angle_unit(target, self.box_angle_unit)
        target.add_field("labels", labels)
        target.add_field("pair_labels", pair_labels)
        target.add_field("relation_triplets", rel_triplets)
        target.add_field("all_relation_triplets", all_rel_triplets)
        target.add_field("image_id", torch.tensor([int(record["image_index"])], dtype=torch.long))
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
        if record["attributes"].size > 0:
            target.add_field("attributes", torch.as_tensor(record["attributes"], dtype=torch.long))
        target = adapt_target_to_task(target, self.task)

        meta = {
            "image_id": int(record["image_index"]),
            "file_name": record["file_name"],
            "height": int(record["height"]),
            "width": int(record["width"]),
            "dataset_name": self.metadata.dataset_name,
            "split": self.split,
        }
        if tile is not None:
            meta["tile_origin"] = (int(tile["x0"]), int(tile["y0"]))
            meta["tile_scale"] = (1.0, 1.0)
            meta["source_image_id"] = int(record.get("_source_image_id", record["image_index"]))
            meta["source_width"] = int(record.get("_source_width", record["width"]))
            meta["source_height"] = int(record.get("_source_height", record["height"]))
            core = record.get("_tile_core", tile)
            meta["tile_core"] = (
                int(core["x0"]),
                int(core["y0"]),
                int(core["x1"]),
                int(core["y1"]),
            )
        return SGGSample(image=image, target=target, meta=meta)
