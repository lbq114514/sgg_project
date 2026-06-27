from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path("toy_data")
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx in range(8):
        w = h = 256
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        boxes = [
            [30, 30, 90, 90],
            [110, 40, 180, 90],
            [80, 140, 170, 210],
        ]
        colors = [(200, 20, 20), (20, 180, 20), (20, 20, 200)]
        objs = []
        for i, (box, color) in enumerate(zip(boxes, colors)):
            draw.rectangle(box, outline=color, width=3)
            objs.append({"bbox": box, "label": i + 1})
        rels = [
            {"subject_id": 0, "object_id": 1, "predicate": 1},
            {"subject_id": 1, "object_id": 2, "predicate": 2},
        ]
        file_name = f"{idx:04d}.png"
        img.save(img_dir / file_name)
        records.append({
            "id": idx,
            "file_name": file_name,
            "width": w,
            "height": h,
            "objects": objs,
            "relations": rels,
        })
    ann = {
        "images": records,
        "categories": {"obj1": 1, "obj2": 2, "obj3": 3},
        "predicates": {"left_of": 1, "overlap_with": 2},
    }
    with (root / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
    print(f"Wrote dataset to {root}")


if __name__ == "__main__":
    main()
