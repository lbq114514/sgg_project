from __future__ import annotations

import argparse
import importlib.util

import torch

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
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
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = get_default_cfg()
    if args.config:
        cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)
    datasets = build_datasets(cfg, splits=("test",))
    test_meta = datasets["test"].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        test_meta.categories[i] for i in sorted(test_meta.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
        test_meta.predicates[i] for i in sorted(test_meta.predicates.keys())
    ]
    model = SceneGraphDetector(cfg)
    dataloaders = build_dataloaders(
        cfg,
        splits=("test",),
        datasets=datasets,
        shuffle_map={"test": False},
    )
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    loader = dataloaders["test"]
    metrics = trainer.evaluate_loader(loader)
    print(metrics)


if __name__ == "__main__":
    main()
