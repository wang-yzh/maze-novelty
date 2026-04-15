from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev


METRICS = [
    "test_success",
    "test_avg_steps",
    "test_stability",
    "test_score",
    "archive_coverage",
    "archive_unique_ratio",
    "success_path_diversity",
    "runtime_seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final metrics across run directories.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    rows = []
    for run_dir in args.runs:
        metrics_path = run_dir / "metrics.csv"
        with metrics_path.open() as handle:
            run_rows = list(csv.DictReader(handle))
        max_generation = max(int(row["generation"]) for row in run_rows)
        for row in run_rows:
            if int(row["generation"]) == max_generation:
                rows.append({"run": run_dir.name, **row})

    methods = sorted({row["method"] for row in rows})
    print("Final rows")
    for row in rows:
        print(
            row["run"],
            row["method"],
            "success=",
            row["test_success"],
            "steps=",
            row["test_avg_steps"],
            "score=",
            row["test_score"],
            "coverage=",
            row["archive_coverage"],
            "unique=",
            row.get("archive_unique_ratio", "n/a"),
            "success_diversity=",
            row.get("success_path_diversity", "n/a"),
            "runtime_seconds=",
            row.get("runtime_seconds", "n/a"),
        )

    print("\nAverages")
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        parts = [method]
        for metric in METRICS:
            values = [float(row.get(metric, 0.0)) for row in method_rows]
            parts.append(f"{metric}={mean(values):.4f}+/-{pstdev(values):.4f}")
        print(" | ".join(parts))

    print("\nNonzero success counts")
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        count = sum(float(row["test_success"]) > 0.0 for row in method_rows)
        print(f"{method}: {count}/{len(method_rows)}")

    print("\nPer-run best by test_score")
    best_rows = []
    for run in sorted({row["run"] for row in rows}):
        run_rows = [row for row in rows if row["run"] == run]
        best = max(run_rows, key=lambda row: float(row["test_score"]))
        best_rows.append(best)
        print(f"{run}: {best['method']} score={best['test_score']}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.csv_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
