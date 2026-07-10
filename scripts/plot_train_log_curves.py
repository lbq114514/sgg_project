#!/usr/bin/env python3
"""Plot training loss / metric curves from this project's plain-text logs.

Examples:
  python scripts/plot_train_log_curves.py \
    outputs/star_predcls_obb_tail_aux/train.log

  python scripts/plot_train_log_curves.py \
    outputs/star_predcls_obb_tail_aux/train.log \
    --out-dir outputs/star_predcls_obb_tail_aux/curves \
    --ks 1500 2000
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TRAIN_LOSS_RE = re.compile(r"^epoch\s+(\d+)/(\d+)\s+train_loss:\s+(\{.*\})\s*$")
LR_RE = re.compile(r"^epoch\s+(\d+)/(\d+)\s+lr:\s+([0-9.eE+-]+)\s*$")
TABLE_METRIC_RE = re.compile(r"^(R|mR|HMR|A|family-mR)\s*\|\s*(.+?)\s*$")
AT_VALUE_RE = re.compile(r"@(\d+)\s+([0-9.eE+-]+)")
SGG_EVAL_RE = re.compile(r"^SGG eval:\s*(.+?)\s*;\s*for mode=.*type=(.+?)\.\s*$")
SGG_PAIR_RE = re.compile(r"\b(R|mR)\s*@\s*(\d+):\s*([0-9.eE+-]+)")
SAVED_BEST_RE = re.compile(r"^Saved best\s+(.+?)\s+checkpoint:\s+(.+?)=([0-9.eE+-]+)\s*$")


def parse_log(path: Path) -> dict[str, Any]:
    losses: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    saved_best: list[dict[str, Any]] = []

    current_epoch: int | None = None

    for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        match = TRAIN_LOSS_RE.match(line)
        if match:
            epoch = int(match.group(1))
            current_epoch = epoch
            try:
                values = ast.literal_eval(match.group(3))
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Failed to parse train_loss at {path}:{lineno}") from exc
            row = {"epoch": epoch, "line": lineno}
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    row[str(key)] = float(value)
            losses.append(row)
            continue

        match = LR_RE.match(line)
        if match:
            epoch = int(match.group(1))
            current_epoch = epoch
            lr_rows.append({"epoch": epoch, "line": lineno, "lr": float(match.group(3))})
            continue

        match = TABLE_METRIC_RE.match(line)
        if match:
            name = match.group(1)
            for k_str, value_str in AT_VALUE_RE.findall(match.group(2)):
                metrics.append(
                    {
                        "epoch": current_epoch,
                        "line": lineno,
                        "metric": name,
                        "k": int(k_str),
                        "value": float(value_str),
                    }
                )
            continue

        match = SGG_EVAL_RE.match(line)
        if match:
            # Original RPCM style:
            # SGG eval: R @ 1000: ...; R @ 1500: ...; ... type=Recall(Main).
            for name, k_str, value_str in SGG_PAIR_RE.findall(match.group(1)):
                metrics.append(
                    {
                        "epoch": current_epoch,
                        "line": lineno,
                        "metric": name,
                        "k": int(k_str),
                        "value": float(value_str),
                    }
                )
            continue

        match = SAVED_BEST_RE.match(line)
        if match:
            saved_best.append(
                {
                    "line": lineno,
                    "name": match.group(1),
                    "quantity": match.group(2),
                    "value": float(match.group(3)),
                    "epoch": current_epoch,
                }
            )

    return {
        "losses": losses,
        "lr": lr_rows,
        "metrics": metrics,
        "saved_best": saved_best,
    }


def write_csvs(parsed: dict[str, Any], out_dir: Path) -> None:
    losses = parsed["losses"]
    if losses:
        keys = ["epoch", "line"] + sorted({k for row in losses for k in row if k not in {"epoch", "line"}})
        with (out_dir / "losses.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(losses)

    lr_rows = parsed["lr"]
    if lr_rows:
        with (out_dir / "lr.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "line", "lr"])
            writer.writeheader()
            writer.writerows(lr_rows)

    metrics = parsed["metrics"]
    if metrics:
        with (out_dir / "metrics_long.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "line", "metric", "k", "value"])
            writer.writeheader()
            writer.writerows(metrics)

        wide: dict[int, dict[str, Any]] = defaultdict(dict)
        for row in metrics:
            epoch = row["epoch"]
            if epoch is None:
                continue
            wide[int(epoch)]["epoch"] = int(epoch)
            wide[int(epoch)][f"{row['metric']}@{row['k']}"] = row["value"]
        columns = ["epoch"] + sorted({key for row in wide.values() for key in row if key != "epoch"})
        with (out_dir / "metrics_wide.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for epoch in sorted(wide):
                writer.writerow(wide[epoch])


def best_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for row in parsed["metrics"]:
        if row["epoch"] is None:
            continue
        key = f"{row['metric']}@{row['k']}"
        if key not in best or row["value"] > best[key]["value"]:
            best[key] = {"epoch": row["epoch"], "value": row["value"], "line": row["line"]}
    return {
        "best_metrics": best,
        "saved_best": parsed["saved_best"],
        "num_loss_rows": len(parsed["losses"]),
        "num_lr_rows": len(parsed["lr"]),
        "num_metric_rows": len(parsed["metrics"]),
    }


def _import_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. CSV files were still written if parsing succeeded."
        ) from exc
    return plt


def plot_losses(parsed: dict[str, Any], out_dir: Path) -> None:
    losses = parsed["losses"]
    if not losses:
        return
    plt = _import_matplotlib()
    epochs = [row["epoch"] for row in losses]

    main_keys = [key for key in ("loss_total", "loss_rel", "pull") if key in losses[-1]]
    small_keys = [key for key in ("loss_tail_aux", "sep") if key in losses[-1]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for key in main_keys:
        axes[0].plot(epochs, [row.get(key, float("nan")) for row in losses], label=key)
    axes[0].set_ylabel("main loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for key in small_keys:
        axes[1].plot(epochs, [row.get(key, float("nan")) for row in losses], label=key)
    axes[1].set_ylabel("aux/small loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    if "grad_norm" in losses[-1]:
        axes[2].plot(epochs, [row.get("grad_norm", float("nan")) for row in losses], label="grad_norm")
        axes[2].legend()
    axes[2].set_ylabel("grad norm")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Training losses")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=160)
    plt.close(fig)


def plot_lr(parsed: dict[str, Any], out_dir: Path) -> None:
    rows = parsed["lr"]
    if not rows:
        return
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot([row["epoch"] for row in rows], [row["lr"] for row in rows], label="lr")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lr_curve.png", dpi=160)
    plt.close(fig)


def plot_metrics(parsed: dict[str, Any], out_dir: Path, metrics: set[str], ks: set[int]) -> None:
    rows = [
        row
        for row in parsed["metrics"]
        if row["epoch"] is not None and row["metric"] in metrics and row["k"] in ks
    ]
    if not rows:
        return
    plt = _import_matplotlib()
    by_series: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        by_series[(row["metric"], row["k"])].append((int(row["epoch"]), float(row["value"])))

    fig, ax = plt.subplots(figsize=(11, 6))
    for metric in sorted(metrics):
        for k in sorted(ks):
            series = sorted(by_series.get((metric, k), []))
            if not series:
                continue
            epochs = [item[0] for item in series]
            values = [item[1] for item in series]
            linestyle = "-" if k == max(ks) else "--"
            ax.plot(epochs, values, linestyle=linestyle, marker="o", markersize=3, label=f"{metric}@{k}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to train.log")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <log parent>/curves",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["R", "mR", "HMR"],
        help="Aggregate metrics to plot, e.g. R mR HMR A family-mR",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1500, 2000], help="Top-K values to plot")
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else log_path.parent / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_log(log_path)
    write_csvs(parsed, out_dir)
    summary = best_summary(parsed)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    plot_losses(parsed, out_dir)
    plot_lr(parsed, out_dir)
    plot_metrics(parsed, out_dir, set(args.metrics), set(args.ks))

    print(f"Wrote curves to: {out_dir}")
    print(json.dumps(summary["best_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
