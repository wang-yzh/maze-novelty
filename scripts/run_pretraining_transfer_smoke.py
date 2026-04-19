from __future__ import annotations
# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

from minigrid_adapter import MiniGridSpec
from pretraining.minigrid_schedules import build_pretrain_artifact
from transfer.evaluator import AdaptationConfig, evaluate_scratch, evaluate_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test pretraining-to-transfer evaluation.")
    parser.add_argument("--source-env", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--target-env", default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="geometry")
    parser.add_argument("--pretrain-episodes", type=int, default=80)
    parser.add_argument("--pretrain-epsilon", type=float, default=0.25)
    parser.add_argument(
        "--pretrain-methods",
        default=(
            "simple_q_pretrain,"
            "operate_replay_pretrain,"
            "cyclic_motif_fast_replay_pretrain,"
            "cyclic_subgoal_ecology_replay_pretrain"
        ),
        help="Comma-separated pretraining methods to compare.",
    )
    parser.add_argument("--adapt-episodes", type=int, default=60)
    parser.add_argument("--adapt-epsilon", type=float, default=0.18)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pretraining_transfer_smoke.csv")
    args = parser.parse_args()

    source_spec = MiniGridSpec(args.source_env, args.max_steps, args.seed, args.state_encoder)
    target_spec = MiniGridSpec(args.target_env, args.max_steps, args.seed, args.state_encoder)
    config = AdaptationConfig(
        episodes=args.adapt_episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        epsilon=args.adapt_epsilon,
        threshold=args.threshold,
    )

    scratch_report, _scratch_points = evaluate_scratch(target_spec, args.seed + 1000, config)
    methods = [method.strip() for method in args.pretrain_methods.split(",") if method.strip()]
    report_rows = []
    for idx, method in enumerate(methods):
        artifact = build_pretrain_artifact(
            method,
            source_spec,
            args.seed + 2000 + idx * 100,
            args.pretrain_episodes,
            args.pretrain_epsilon,
        )
        report, _points = evaluate_transfer(
            artifact,
            target_spec,
            args.seed + 3000 + idx * 100,
            config,
            scratch_final_score=scratch_report.final_score,
        )
        report_rows.append(_report_row(report, artifact.metadata))

    rows = [_report_row(scratch_report, {}), *report_rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["method"],
            "zero_shot=",
            round(row["zero_shot_score"], 4),
            "auc=",
            round(row["adaptation_auc"], 4),
            "first_success=",
            row["time_to_first_success"],
            "final=",
            round(row["final_score"], 4),
            "lift=",
            round(row["transfer_lift"], 4),
            "negative=",
            row["negative_transfer"],
        )
    print(f"Wrote {args.output}")


def _report_row(report, metadata: dict) -> dict:
    row = asdict(report)
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


def _fieldnames(rows: list[dict]) -> list[str]:
    ordered = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


if __name__ == "__main__":
    main()
