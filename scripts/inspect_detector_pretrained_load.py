from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import h5py  # noqa: F401
except ModuleNotFoundError:
    h5py_stub = types.ModuleType("h5py")

    def _missing_h5py(*args, **kwargs):
        raise ModuleNotFoundError("h5py is required for dataset loading but not installed in this environment.")

    h5py_stub.File = _missing_h5py
    sys.modules["h5py"] = h5py_stub

try:
    import mmrotate.core  # noqa: F401
except ModuleNotFoundError:
    mmrotate_stub = types.ModuleType("mmrotate")
    mmrotate_core_stub = types.ModuleType("mmrotate.core")

    def _missing_mmrotate(*args, **kwargs):
        raise ModuleNotFoundError("mmrotate is required for OBB geometry ops but not installed in this environment.")

    mmrotate_core_stub.obb2xyxy = _missing_mmrotate
    mmrotate_core_stub.obb2poly = _missing_mmrotate
    mmrotate_core_stub.poly2obb = _missing_mmrotate
    mmrotate_stub.core = mmrotate_core_stub
    sys.modules["mmrotate"] = mmrotate_stub
    sys.modules["mmrotate.core"] = mmrotate_core_stub

from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector


def apply_runtime_cfg(cfg: dict) -> None:
    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def load_py_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def load_label_info(dict_file: Path) -> tuple[list[str], list[str]]:
    info = json.loads(dict_file.read_text(encoding="utf-8"))
    label_to_idx = dict(info["label_to_idx"])
    predicate_to_idx = dict(info["predicate_to_idx"])
    label_to_idx.setdefault("__background__", 0)
    predicate_to_idx.setdefault("__background__", 0)
    class_names = sorted(label_to_idx, key=lambda key: label_to_idx[key])
    relation_names = sorted(predicate_to_idx, key=lambda key: predicate_to_idx[key])
    return class_names, relation_names


def sync_model_names_from_dataset_dict(cfg: dict) -> None:
    datasets_cfg = cfg.get("DATASETS", {})
    split_cfg = None
    for split_name in ("TRAIN", "VAL", "TEST"):
        candidate = datasets_cfg.get(split_name, {})
        if isinstance(candidate, dict) and candidate.get("DICT_FILE"):
            split_cfg = candidate
            break
    if split_cfg is None:
        return

    dict_file = Path(str(split_cfg["DICT_FILE"])).expanduser()
    if not dict_file.is_file():
        raise FileNotFoundError(f"Dataset DICT_FILE not found: {dict_file.resolve()}")

    class_names, relation_names = load_label_info(dict_file)
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = class_names
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = relation_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SceneGraphDetector from config and inspect pretrained detector loading."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to python config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional override for MODEL.PRETRAINED_DETECTOR.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used after model construction. Default: cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_py_config(str(config_path))
    apply_runtime_cfg(cfg)
    sync_model_names_from_dataset_dict(cfg)

    configured_checkpoint = str(cfg.get("MODEL", {}).get("PRETRAINED_DETECTOR", ""))
    checkpoint_path = args.checkpoint or configured_checkpoint
    if not checkpoint_path:
        raise ValueError("No pretrained checkpoint specified. Set MODEL.PRETRAINED_DETECTOR or pass --checkpoint.")
    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_file.resolve()}")
    cfg["MODEL"]["PRETRAINED_DETECTOR"] = str(checkpoint_file)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but device={args.device} was requested.")

    print(f"Config: {config_path}", flush=True)
    print(f"Pretrained detector checkpoint: {checkpoint_file.resolve()}", flush=True)
    print("Building model and triggering detector pretrained load...", flush=True)

    model = SceneGraphDetector(cfg)
    model.to(torch.device(args.device))
    model.eval()

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        f"Model build complete on device={args.device}. "
        f"total_params={total_params}, trainable_params={trainable_params}",
        flush=True,
    )


if __name__ == "__main__":
    main()
