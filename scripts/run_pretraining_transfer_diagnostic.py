from __future__ import annotations
# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

from minigrid_adapter import MiniGridSpec
from pretraining.minigrid_schedules import build_pretrain_artifact
from transfer.evaluator import AdaptationConfig, evaluate_scratch, evaluate_transfer


DEFAULT_METHODS = (
    "simple_q_pretrain,"
    "operate_replay_pretrain,"
    "cyclic_motif_fast_replay_pretrain,"
    "cyclic_subgoal_ecology_replay_pretrain"
)

SUMMARY_METRICS = [
    "zero_shot_score",
    "adaptation_auc",
    "time_to_first_success",
    "time_to_threshold",
    "final_score",
    "transfer_lift",
    "artifact_source_score",
    "artifact_source_success",
    "artifact_train_successes",
    "artifact_replay_bank_size",
    "artifact_motif_count",
    "artifact_active_niches",
    "artifact_subgoal_motif_count",
    "artifact_transition_motif_count",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed pretraining transfer diagnostic.")
    parser.add_argument("--source-env", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--target-env", default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="geometry")
    parser.add_argument("--pretrain-episodes", type=int, default=40)
    parser.add_argument("--pretrain-epsilon", type=float, default=0.25)
    parser.add_argument("--adapt-episodes", type=int, default=30)
    parser.add_argument("--adapt-epsilon", type=float, default=0.18)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pretraining_transfer_diagnostic.csv")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "outputs" / "pretraining_transfer_diagnostic_summary.csv",
    )
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        source_spec = MiniGridSpec(args.source_env, args.max_steps, seed, args.state_encoder)
        target_spec = MiniGridSpec(args.target_env, args.max_steps, seed, args.state_encoder)
        config = AdaptationConfig(
            episodes=args.adapt_episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            epsilon=args.adapt_epsilon,
            threshold=args.threshold,
        )
        scratch_report, _scratch_points = evaluate_scratch(target_spec, seed + 1000, config)
        scratch_row = _report_row(scratch_report, {}, seed)
        rows.append(scratch_row)
        print(_format_progress(scratch_row))

        for idx, method in enumerate(methods):
            artifact = build_pretrain_artifact(
                method,
                source_spec,
                seed + 2000 + idx * 100,
                args.pretrain_episodes,
                args.pretrain_epsilon,
            )
            report, _points = evaluate_transfer(
                artifact,
                target_spec,
                seed + 3000 + idx * 100,
                config,
                scratch_final_score=scratch_report.final_score,
            )
            row = _report_row(report, artifact.metadata, seed)
            rows.append(row)
            print(_format_progress(row))

    summary_rows = _summary_rows(rows)
    _write_csv(args.output, rows)
    _write_csv(args.summary_output, summary_rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


def _report_row(report, metadata: dict[str, Any], experiment_seed: int) -> dict[str, Any]:
    row = asdict(report)
    row["experiment_seed"] = experiment_seed
    row["artifact_source_score"] = metadata.get("source_score", "")
    row["artifact_source_success"] = metadata.get("source_success", "")
    row["artifact_train_successes"] = metadata.get("train_successes", "")
    row["artifact_replay_bank_size"] = metadata.get("replay_bank_size", "")
    row["artifact_motif_count"] = metadata.get("motif_count", "")
    row["artifact_active_niches"] = metadata.get("active_niches", "")
    row["artifact_subgoal_motif_count"] = metadata.get("subgoal_motif_count", "")
    row["artifact_transition_motif_count"] = metadata.get("transition_motif_count", "")
    row["artifact_metadata_json"] = json.dumps(metadata, sort_keys=True)
    return row


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    output = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "runs": len(method_rows),
            "nonzero_final_count": sum(_float(row["final_score"]) > 0.0 for row in method_rows),
            "negative_transfer_count": sum(str(row["negative_transfer"]) == "True" for row in method_rows),
        }
        for metric in SUMMARY_METRICS:
            values = [_float(row.get(metric, "")) for row in method_rows if row.get(metric, "") != ""]
            summary[f"{metric}_mean"] = mean(values) if values else ""
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    ordered = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _float(value: object) -> float:
    if not isinstance(value, int | float | str):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"seed={row['experiment_seed']} {row['method']} "
        f"final={_float(row['final_score']):.4f} "
        f"auc={_float(row['adaptation_auc']):.4f} "
        f"lift={_float(row['transfer_lift']):.4f} "
        f"artifact_score={_float(row['artifact_source_score']):.4f} "
        f"motifs={row['artifact_motif_count']}"
    )


if __name__ == "__main__":
    main()
