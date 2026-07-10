#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


CURRENT_ROOT = Path(__file__).resolve().parents[1]
# SGG-ToolKit source/config is the parity reference.  Its runnable Python 3.11
# extension is maintained in the RPCM checkout, so use that runtime by default.
# Set SGG_REFERENCE_ROOT to execute another compatible checkout explicitly.
REFERENCE_ROOT = Path(
    os.environ.get("SGG_REFERENCE_ROOT", "/home/ubuntu/research/ssd/RPCM")
)
if str(CURRENT_ROOT) not in sys.path:
    sys.path.append(str(CURRENT_ROOT))

import torch
from torch import nn

from scripts.rpcm_probe_utils import HookDumper, summarize_value, tensor_digest


ORIGINAL_OPTS_BASE = [
    "MODEL.ROI_RELATION_HEAD.USE_GT_BOX",
    "True",
    "MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS",
    "False",
    "MODEL.ROI_RELATION_HEAD.BIAS_LAMBDA",
    "0.2",
    "MODEL.ROI_RELATION_HEAD.PREDICTOR",
    "RPCM",
    "DTYPE",
    "float32",
    "GLOVE_DIR",
    "glove",
    "SOLVER.IMS_PER_BATCH",
    "16",
    "TEST.IMS_PER_BATCH",
    "1",
    "MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE",
    "512",
    "Type",
    "Large_RS_OBB",
    "filter_method",
    "PPG",
    "OUTPUT_DIR",
    "/tmp/rpcm_probe_original",
]


def original_opts_for_task(task: str) -> list[str]:
    task = str(task).lower()
    if task not in {"predcls", "sgcls"}:
        raise ValueError(f"Unsupported probe task {task!r}; expected predcls or sgcls.")
    opts = list(ORIGINAL_OPTS_BASE)
    opts.extend(
        [
            "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL",
            "True" if task == "predcls" else "False",
        ]
    )
    return opts


def find_batch(loader, image_id: int):
    for batch in loader:
        image_ids = batch[2]
        if int(image_ids[0]) == int(image_id):
            return batch
    raise ValueError(f"image_id={image_id} not found")


def get_relation_tensor(target):
    for name in ("relation_tuple", "relation_triplets", "relations"):
        if target.has_field(name):
            return target.get_field(name)
    if target.has_field("relation"):
        rel_mat = target.get_field("relation")
        nz = (rel_mat > 0).nonzero(as_tuple=False)
        if nz.numel() == 0:
            return rel_mat.new_zeros((0, 3))
        return torch.cat([nz, rel_mat[nz[:, 0], nz[:, 1]].view(-1, 1)], dim=1)
    return torch.zeros((0, 3), dtype=torch.long)


def import_reference_runtime(reference_root: Path):
    for path in (reference_root, reference_root / "mmrote_RS"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmrotate.models import build_detector
        from maskrcnn_benchmark.config import cfg
        from maskrcnn_benchmark.data import make_data_loader
    except ImportError as error:
        py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        raise RuntimeError(
            f"Cannot import the reference runtime at {reference_root} under {py_tag}. "
            "Select the compiled RPCM checkout or build maskrcnn_benchmark._C for this "
            "Python version, then pass it through --reference-root."
        ) from error
    return Config, load_checkpoint, build_detector, cfg, make_data_loader


class DetectorOnlyROIHeads(nn.Module):
    """Stop the original detector immediately before relation prediction."""

    def forward(self, features, proposals, targets=None, logger=None, **kwargs):
        del targets, logger, kwargs
        return features, proposals, {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one-image original RPCM intermediate tensors.")
    parser.add_argument(
        "--reference-root",
        "--rpcm-root",
        dest="reference_root",
        default=str(REFERENCE_ROOT),
        help="Reference runtime root; --rpcm-root remains a compatibility alias.",
    )
    parser.add_argument("--config-file", default="configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml")
    parser.add_argument("--mm-config", default="configs/RSOBB/STAR_obb_predcls_sgcls.py")
    parser.add_argument("--checkpoint", default="/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth")
    parser.add_argument("--task", default="predcls", choices=("predcls", "sgcls"))
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Stop after detector object logits/features; do not execute the relation predictor.",
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="For sgcls, explicitly execute the original relation predictor after detector probing.",
    )
    parser.add_argument("--image-id", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="/home/ubuntu/research/ssd/sgg_project/outputs/rpcm_probe/original_image4.pt")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-cols", type=int, default=64)
    parser.add_argument("--max-flat", type=int, default=4096)
    args = parser.parse_args()
    if args.detector_only and args.full_model:
        parser.error("--detector-only and --full-model cannot be used together")

    # The historical PredCls checkpoints commonly used for parity probing do
    # not contain the sgcls-only ``out_obj`` branch.  The first sgcls parity
    # milestone is detector semantics (cls_score / labels / RoI features), so
    # make that safe path the default.  Full predictor execution stays
    # available for a checkpoint known to contain all sgcls predictor weights.
    detector_only = bool(args.detector_only or (args.task == "sgcls" and not args.full_model))

    reference_root = Path(args.reference_root).resolve()
    os.chdir(reference_root)
    Config, load_checkpoint, build_detector, cfg, make_data_loader = import_reference_runtime(reference_root)
    Path("/tmp/rpcm_probe_original").mkdir(parents=True, exist_ok=True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(original_opts_for_task(args.task))
    cfg.MODEL.DEVICE = args.device
    cfg_mmcv = Config.fromfile(args.mm_config)
    cfg["mmcv"] = cfg_mmcv.data.test
    cfg_mmcv.model["ori_cfg"] = cfg

    loaders = make_data_loader(cfg, mode="test", is_distributed=False)
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
    images, targets, image_ids, img_filename, imgs, tar1 = find_batch(loader, args.image_id)

    model = build_detector(
        cfg_mmcv.model,
        train_cfg=cfg_mmcv.get("train_cfg"),
        test_cfg=cfg_mmcv.get("test_cfg"),
    )
    device = torch.device(args.device)
    model.to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    model.eval()

    original_roi_heads = model.roi_heads
    rel_head = original_roi_heads.relation
    predictor = rel_head.predictor
    if detector_only:
        model.roi_heads = DetectorOnlyROIHeads()
    hooks = HookDumper(max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat)
    hooks.add(model.backbone, "backbone")
    hooks.add(model.neck, "neck")
    hooks.add(model.roi_head.bbox_roi_extractor, "roi_head.bbox_roi_extractor")
    hooks.add(model.roi_head.bbox_head, "roi_head.bbox_head")
    if not detector_only:
        hooks.add(rel_head.union_feature_extractor, "relation.union_feature_extractor")
        hooks.add(predictor.pairwise_feature_extractor, "predictor.pairwise_feature_extractor")
        hooks.add(predictor.down_samp, "predictor.down_samp")
        hooks.add(predictor.rel_residual, "predictor.rel_residual")
        hooks.add(predictor.rel_norm, "predictor.rel_norm")
        hooks.add(predictor.rel_proto, "predictor.rel_proto")

    with torch.inference_mode():
        output = model(
            images.to(device),
            [target.to(device) for target in targets],
            logger=None,
            sgd_data=[imgs, tar1] if imgs is not None else None,
        )
    hooks.close()
    pred = output[0].to("cpu")
    target_cpu = targets[0]
    target_relations = get_relation_tensor(target_cpu).cpu()

    result_fields = {}
    for field in (
        "base_rel_pair_idxs",
        "sema_rel_pair_idxs",
        "final_rel_pair_idxs",
        "pruned_rel_pair_idxs",
        "pred_rel_scores",
        "pred_rel_labels",
        "rel_pair_idxs",
    ):
        if pred.has_field(field):
            result_fields[field] = summarize_value(
                pred.get_field(field),
                max_rows=args.max_rows,
                max_cols=args.max_cols,
                max_flat=args.max_flat,
            )
    object_fields = {}
    for field in (
        "labels",
        "gt_labels",
        "predict_logits",
        "pred_labels",
        "pred_scores",
    ):
        if pred.has_field(field):
            object_fields[field] = summarize_value(
                pred.get_field(field),
                max_rows=args.max_rows,
                max_cols=args.max_cols,
                max_flat=args.max_flat,
            )

    dump = {
        "source": "original",
        "reference_root": str(reference_root),
        "config_file": args.config_file,
        "mm_config": args.mm_config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": args.task,
        "detector_only": detector_only,
        "image_id": int(args.image_id),
        "image_ids": list(image_ids),
        "image_filename": img_filename[0] if img_filename else "",
        "image": tensor_digest(images.tensors.cpu(), max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "target_size": tuple(target_cpu.size),
        "target_boxes": tensor_digest(target_cpu.bbox.cpu(), max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "target_labels": tensor_digest(target_cpu.get_field("labels").cpu(), max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "target_fields": list(target_cpu.extra_fields.keys()),
        "target_relations": tensor_digest(target_relations, max_rows=args.max_rows, max_cols=args.max_cols, max_flat=args.max_flat),
        "hooks": hooks.records,
        "object_fields": object_fields,
        "result_fields": result_fields,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
