from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
from sgg.data.statistics import predicate_histogram
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector


def apply_runtime_cfg(cfg):
    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-rpcm", type=str, default="",
        help="Initialize compatible TypedHyperRPCM blocks from an RPCM checkpoint.",
    )
    parser.add_argument(
        "--init-typed", type=str, default="",
        help="Initialize from a compatible TypedHyperRPCM checkpoint while allowing newly added heads.",
    )
    parser.add_argument("--start-epoch", type=int, default=None, help="Epoch index to resume from, 0-based. Defaults to checkpoint epoch.")
    parser.add_argument("--resume-step", type=int, default=None, help="Override global training step after loading a checkpoint.")
    args = parser.parse_args()

    cfg = get_default_cfg()
    if args.config:
        user_cfg = load_py_config(args.config)
        cfg = user_cfg
    apply_runtime_cfg(cfg)
    datasets = build_datasets(cfg)
    if "train" in datasets:
        train_meta = datasets["train"].metadata
        cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
            train_meta.categories[i] for i in sorted(train_meta.categories.keys())
        ]
        cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
            train_meta.predicates[i] for i in sorted(train_meta.predicates.keys())
        ]
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        if not rel_cfg.get("PREDICATE_COUNTS"):
            rel_cfg["PREDICATE_COUNTS"] = predicate_histogram(datasets["train"])
    model = SceneGraphDetector(cfg)
    # print(model)
    dataloaders = build_dataloaders(
        cfg,
        datasets=datasets,
        shuffle_map={"train": True, "val": False, "test": False},
    )
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    start_epoch = 0
    init_modes = sum(bool(value) for value in (args.init_rpcm, args.init_typed, args.resume))
    if init_modes > 1:
        raise ValueError("--init-rpcm, --init-typed and --resume are mutually exclusive")
    if args.init_rpcm:
        trainer.load_rpcm_predictor_weights(args.init_rpcm)
    if args.init_typed:
        trainer.load_typed_stage_weights(args.init_typed)
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
    if args.start_epoch is not None:
        start_epoch = int(args.start_epoch)
    if args.resume_step is not None:
        trainer.global_step = int(args.resume_step)
    if args.resume or args.start_epoch is not None or args.resume_step is not None:
        print(
            f"Resume state: checkpoint={args.resume or '<none>'}, start_epoch={start_epoch}, global_step={trainer.global_step}",
            flush=True,
        )
    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
