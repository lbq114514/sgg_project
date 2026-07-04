from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run one evaluation pass on val/test split.")
    parser.add_argument("--config", type=str, required=True, help="Path to python config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON file to save metrics.",
    )
    parser.add_argument(
        "--filter-method",
        choices=("NONE", "PPG", "PPN"),
        default=None,
        help="Optional runtime override for the configured pair filter.",
    )
    parser.add_argument(
        "--pair-filter-checkpoint",
        default="",
        help="Checkpoint override for the selected PPG/PPN filter.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_default_cfg()
    cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    if args.filter_method is not None:
        rel_cfg["TEST_FILTER_METHOD"] = args.filter_method
    if args.pair_filter_checkpoint:
        if str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper() == "PPN":
            rel_cfg["PPN_MODEL_PATH"] = args.pair_filter_checkpoint
        elif str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper() == "PPG":
            rel_cfg["PPG_MODEL_PATH_OBB"] = args.pair_filter_checkpoint
        else:
            raise ValueError("--pair-filter-checkpoint requires --filter-method PPN or PPG")
    filter_method = str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper()
    filter_path = (
        rel_cfg.get("PPN_MODEL_PATH", "")
        if filter_method == "PPN"
        else rel_cfg.get("PPG_MODEL_PATH_OBB", "")
        if filter_method == "PPG"
        else ""
    )
    print(
        f"Resolved pair filter: method={filter_method}, checkpoint={filter_path or '<none>'}",
        flush=True,
    )

    split = args.split.lower()
    datasets = build_datasets(cfg, splits=(split,))
    split_meta = datasets[split].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        split_meta.categories[i] for i in sorted(split_meta.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
        split_meta.predicates[i] for i in sorted(split_meta.predicates.keys())
    ]
    dataloaders = build_dataloaders(
        cfg,
        splits=(split,),
        datasets=datasets,
        shuffle_map={split: False},
    )

    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    trainer.load_checkpoint(args.checkpoint)

    metrics, _ = trainer.evaluate_loader(dataloaders[split], return_result=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "config": str(Path(args.config).resolve()),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "split": split,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
