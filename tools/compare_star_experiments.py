#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


DEFAULT_LOGS = {
    "main": "star_train_main.log",
    "pairness_score": "star_train_pairness_score.log",
    "no_proto_bias": "star_train_no_proto_bias.log",
    "pairness_tight": "star_train_pairness_tight.log",
}

EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)\s+train_loss:")


def _metric_at(metrics: dict, name: str, k: int):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get(k, value.get(str(k)))
    return value


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_log(path: Path, recall_k: int):
    current_epoch = None
    rows = []
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        if not line.startswith("Validation:"):
            continue
        payload = line.split("Validation:", 1)[1].strip()
        try:
            metrics = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        rows.append((current_epoch, metrics))
    if not rows:
        return None

    def best_by(metric_name: str):
        return max(
            rows,
            key=lambda item: float(_metric_at(item[1], metric_name, recall_k) or float("-inf")),
        )

    best_r = best_by("R")
    best_mr = best_by("mR")
    latest = rows[-1]
    return {
        "path": str(path),
        "latest_epoch": latest[0],
        "latest": latest[1],
        "best_R_epoch": best_r[0],
        "best_R": best_r[1],
        "best_mR_epoch": best_mr[0],
        "best_mR": best_mr[1],
    }


def print_table(results: dict, recall_k: int):
    headers = [
        "experiment",
        "latest_ep",
        f"latest_R@{recall_k}",
        f"latest_mR@{recall_k}",
        f"latest_HR@{recall_k}",
        f"best_R@{recall_k}",
        "best_R_ep",
        f"best_mR@{recall_k}",
        "best_mR_ep",
        "pair_R",
        "pair_P",
        "gt_pair_cls_R",
        "gt_pair_cls_mR",
        "gt_pair_cov",
    ]
    table = [headers]
    for name, result in results.items():
        if result is None:
            table.append([name, "missing", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        latest = result["latest"]
        best_r = result["best_R"]
        best_mr = result["best_mR"]
        table.append(
            [
                name,
                result["latest_epoch"],
                _fmt(_metric_at(latest, "R", recall_k)),
                _fmt(_metric_at(latest, "mR", recall_k)),
                _fmt(_metric_at(latest, "HR", recall_k)),
                _fmt(_metric_at(best_r, "R", recall_k)),
                result["best_R_epoch"],
                _fmt(_metric_at(best_mr, "mR", recall_k)),
                result["best_mR_epoch"],
                _fmt(latest.get("pair_R")),
                _fmt(latest.get("pair_P", latest.get("pair_frac"))),
                _fmt(_metric_at(latest, "gt_pair_cls_R", recall_k)),
                _fmt(_metric_at(latest, "gt_pair_cls_mR", recall_k)),
                _fmt(latest.get("gt_pair_coverage")),
            ]
        )
    widths = [max(len(str(row[idx])) for row in table) for idx in range(len(headers))]
    for idx, row in enumerate(table):
        print("  ".join(str(cell).ljust(widths[col]) for col, cell in enumerate(row)))
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2000, help="Recall@K to compare.")
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Extra log in name=path form. Can be repeated.",
    )
    args = parser.parse_args()

    logs = dict(DEFAULT_LOGS)
    for item in args.log:
        if "=" not in item:
            raise SystemExit(f"--log must be name=path, got: {item}")
        name, path = item.split("=", 1)
        logs[name] = path

    results = {
        name: parse_log(Path(path), recall_k=args.k)
        for name, path in logs.items()
    }
    print_table(results, recall_k=args.k)


if __name__ == "__main__":
    main()
