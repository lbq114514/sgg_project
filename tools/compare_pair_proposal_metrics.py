from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare PPN, PPG and no-filter pair recall.")
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics]
    topks = sorted(
        {int(k) for run in runs for k in run["metrics"]["recall"]},
    )
    rows = []
    for run in runs:
        metrics = run["metrics"]
        row = {
            "method": run.get("method", "unknown"),
            "checkpoint": run.get("checkpoint", ""),
            "semantic_pairs": metrics["semantic_pairs"],
        }
        for k in topks:
            row[f"R@{k}"] = metrics["recall"].get(str(k))
            row[f"pairs@{k}"] = metrics["avg_candidates"].get(str(k))
        rows.append(row)

    headers = ["method"] + [f"R@{k}" for k in topks] + [f"pairs@{k}" for k in topks]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        values = [row["method"]]
        values.extend(f"{row[f'R@{k}']:.4f}" for k in topks)
        values.extend(f"{row[f'pairs@{k}']:.1f}" for k in topks)
        print(" | ".join(values))
    if args.output:
        Path(args.output).write_text(
            json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
