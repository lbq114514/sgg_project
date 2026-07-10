#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.data.build import build_dataloaders, build_datasets
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from tools.eval_once import apply_runtime_cfg, load_model_only_checkpoint, load_py_config


def _cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


class MemoryRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, name: str, **extra: Any) -> None:
        row: dict[str, Any] = {"name": name}
        row.update(_cuda_mem())
        row.update(extra)
        self.records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def hook(self, name: str):
        def _hook(_module, _inputs, output):
            extra = {}
            if torch.is_tensor(output):
                extra["output_shape"] = list(output.shape)
                extra["output_mb"] = output.numel() * output.element_size() / 1024**2
            elif isinstance(output, (list, tuple)):
                shapes = []
                total = 0.0
                for item in output:
                    if torch.is_tensor(item):
                        shapes.append(list(item.shape))
                        total += item.numel() * item.element_size() / 1024**2
                    elif isinstance(item, dict):
                        for value in item.values():
                            if torch.is_tensor(value):
                                shapes.append(list(value.shape))
                                total += value.numel() * value.element_size() / 1024**2
                if shapes:
                    extra["output_shapes"] = shapes[:8]
                    extra["output_mb"] = total
            self.record(f"after_{name}", **extra)

        return _hook


def _tensor_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 1024**2


def _install_hooks(model: torch.nn.Module, recorder: MemoryRecorder) -> list[Any]:
    handles: list[Any] = []

    def add(module: torch.nn.Module | None, name: str) -> None:
        if module is None:
            return
        handles.append(module.register_forward_hook(recorder.hook(name)))

    add(getattr(model, "backbone", None), "backbone")
    add(getattr(model, "neck", None), "neck")
    roi_head = getattr(model, "roi_head", None)
    if roi_head is not None:
        add(getattr(roi_head, "bbox_roi_extractor", None), "bbox_roi_extractor")
        bbox_head = getattr(roi_head, "bbox_head", None)
        add(bbox_head, "bbox_head")
    rel_head = getattr(getattr(model, "roi_heads", None), "relation", None)
    if rel_head is not None:
        add(getattr(rel_head, "ppg", None), "pair_filter_ppg")
        add(getattr(rel_head, "union_feature_extractor", None), "union_feature_extractor")
        predictor = getattr(rel_head, "predictor", None)
        if predictor is not None:
            add(getattr(predictor, "pairwise_feature_extractor", None), "pairwise_feature_extractor")
            for idx, layer in enumerate(getattr(predictor, "gcn_ent2ent", [])):
                add(layer, f"gcn_ent2ent_{idx}")
            for idx, layer in enumerate(getattr(predictor, "gcn_ent2rel", [])):
                add(layer, f"gcn_ent2rel_{idx}")
            for idx, layer in enumerate(getattr(predictor, "gcn_rel2rel", [])):
                add(layer, f"gcn_rel2rel_{idx}")
            add(getattr(predictor, "down_samp", None), "down_samp")
            add(getattr(predictor, "rel_residual", None), "rel_residual")
            add(getattr(predictor, "rel_norm", None), "rel_norm")
            add(getattr(predictor, "rel_proto", None), "rel_proto")
    return handles


def _patch_get_map_idxs(model: torch.nn.Module, recorder: MemoryRecorder):
    rel_head = getattr(getattr(model, "roi_heads", None), "relation", None)
    predictor = getattr(rel_head, "predictor", None) if rel_head is not None else None
    if predictor is None or not hasattr(predictor, "_get_map_idxs"):
        return None
    original = predictor._get_map_idxs

    def wrapped_get_map_idxs(proposals, proposal_pairs):
        obj_count = sum(len(proposal) for proposal in proposals)
        rel_counts = [int(pair_idx.size(0)) for pair_idx in proposal_pairs]
        rel_count = int(sum(rel_counts))
        dense_rr_mb = (rel_count * rel_count * 4) / 1024**2
        dense_or_mb = (obj_count * rel_count * 4) / 1024**2
        recorder.record(
            "before_get_map_idxs",
            obj_count=int(obj_count),
            rel_count=int(rel_count),
            rel_counts=rel_counts,
            one_rel_rel_float32_mb=dense_rr_mb,
            two_rel_rel_float32_mb=2 * dense_rr_mb,
            one_obj_rel_float32_mb=dense_or_mb,
        )
        out = original(proposals, proposal_pairs)
        names = (
            "subj_pred_map",
            "obj_pred_map",
            "pred_pred_subj",
            "pred_pred_obj",
            "entity_map",
        )
        tensors = {
            name: {
                "shape": list(tensor.shape),
                "mb": _tensor_mb(tensor),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in zip(names, out)
            if torch.is_tensor(tensor)
        }
        recorder.record("after_get_map_idxs", maps=tensors)
        return out

    predictor._get_map_idxs = wrapped_get_map_idxs
    return original


def _restore_get_map_idxs(model: torch.nn.Module, original) -> None:
    if original is None:
        return
    rel_head = getattr(getattr(model, "roi_heads", None), "relation", None)
    predictor = getattr(rel_head, "predictor", None) if rel_head is not None else None
    if predictor is not None:
        predictor._get_map_idxs = original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CUDA memory for one eval batch.")
    parser.add_argument("--config", default="configs/star_predcls_obb_train.py")
    parser.add_argument("--checkpoint", default="/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth")
    parser.add_argument("--checkpoint-load-mode", choices=("full", "model-only", "legacy-rpcm"), default="legacy-rpcm")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--filter-method", choices=("PPG", "PPN", "RSGP"), default=None)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)
    cfg["DATALOADER"]["NUM_WORKERS"] = 0
    if args.filter_method is not None:
        cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"] = args.filter_method

    split = args.split.lower()
    datasets = build_datasets(cfg, splits=(split,))
    meta = datasets[split].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [meta.categories[i] for i in sorted(meta.categories)]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [meta.predicates[i] for i in sorted(meta.predicates)]
    dataloaders = build_dataloaders(cfg, splits=(split,), datasets=datasets, shuffle_map={split: False})

    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    recorder = MemoryRecorder()
    recorder.record("start")
    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=str(device), dataloaders=dataloaders)
    if args.checkpoint_load_mode == "full":
        trainer.load_checkpoint(args.checkpoint)
    elif args.checkpoint_load_mode == "model-only":
        load_model_only_checkpoint(trainer, args.checkpoint, legacy_rpcm=False)
    else:
        load_model_only_checkpoint(trainer, args.checkpoint, legacy_rpcm=True)

    model = trainer.model.eval()
    recorder.record("after_model_load")
    hooks = _install_hooks(model, recorder)
    original_get_map_idxs = _patch_get_map_idxs(model, recorder)

    with torch.inference_mode():
        for batch_idx, (images, targets, metas) in enumerate(dataloaders[split]):
            if batch_idx < args.batch_index:
                continue
            recorder.record(
                "before_batch_to_device",
                batch_idx=int(batch_idx),
                image_shape=list(images.shape),
                target_lengths=[int(len(target)) for target in targets],
                image_ids=[int(meta.get("image_id", -1)) for meta in metas],
            )
            images = images.to(device)
            targets = trainer._move_targets(targets)
            recorder.record("after_batch_to_device")
            preds = model(images, targets if cfg["MODEL"]["TASK"] != "sgdet" else None)
            recorder.record(
                "after_model_forward",
                pred_lengths=[int(len(pred)) for pred in preds],
                pred_pair_counts=[
                    int(pred.get_field("rel_pair_idxs").size(0)) if pred.has_field("rel_pair_idxs") else -1
                    for pred in preds
                ],
            )
            preds_cpu = [pred.to("cpu") for pred in preds]
            recorder.record("after_preds_to_cpu", pred_lengths=[int(len(pred)) for pred in preds_cpu])
            break

    _restore_get_map_idxs(model, original_get_map_idxs)
    for handle in hooks:
        handle.remove()
    recorder.record("done")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(recorder.records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
