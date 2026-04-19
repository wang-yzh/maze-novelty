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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from minigrid_diagnostics import summarize_rollouts
from pretraining.minigrid_schedules import build_pretrain_artifact
from pretraining.minigrid_schedules import run_minigrid_episode
from transfer.evaluator import AdaptationConfig, evaluate_scratch, evaluate_transfer
from transfer.evaluator import TargetReuseConfig, evaluate_transfer_with_target_reuse


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
    "target_probe_subgoal_score",
    "target_probe_region_transitions",
    "target_probe_new_regions",
    "target_probe_mobility",
    "target_probe_success",
    "target_reuse_item_count",
    "target_reuse_avg_item_score",
    "target_reuse_best_item_score",
    "target_reuse_avg_prior_support",
    "target_reuse_best_prior_support",
    "target_reuse_matched_prior_count",
    "target_reuse_executed_prior_count",
    "target_reuse_executed_prior_steps",
    "target_reuse_executed_episode_count",
    "target_reuse_idle_episode_count",
    "target_reuse_executed_episode_avg_subgoal_score",
    "target_reuse_executed_episode_avg_region_transitions",
    "target_reuse_executed_episode_avg_mobility",
    "target_reuse_idle_episode_avg_subgoal_score",
    "target_reuse_idle_episode_avg_region_transitions",
    "target_reuse_idle_episode_avg_mobility",
    "target_reuse_probe_success",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed pretraining transfer diagnostic.")
    parser.add_argument("--source-env", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--target-env", default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument(
        "--target-envs",
        default="",
        help="Optional comma-separated target ladder. Overrides --target-env when set.",
    )
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
    parser.add_argument("--include-target-reuse", action="store_true")
    parser.add_argument("--reuse-match-modes", default="strict")
    parser.add_argument("--reuse-probe-episodes", type=int, default=6)
    parser.add_argument("--reuse-reinforce-passes", type=int, default=4)
    parser.add_argument("--reuse-reward", type=float, default=0.075)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pretraining_transfer_diagnostic.csv")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "outputs" / "pretraining_transfer_diagnostic_summary.csv",
    )
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    target_envs = [env.strip() for env in args.target_envs.split(",") if env.strip()] or [args.target_env]
    reuse_match_modes = [mode.strip() for mode in args.reuse_match_modes.split(",") if mode.strip()]
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        source_spec = MiniGridSpec(args.source_env, args.max_steps, seed, args.state_encoder)
        artifacts = []
        for idx, method in enumerate(methods):
            artifact = build_pretrain_artifact(
                method,
                source_spec,
                seed + 2000 + idx * 100,
                args.pretrain_episodes,
                args.pretrain_epsilon,
            )
            artifacts.append((idx, artifact))

        for target_idx, target_env in enumerate(target_envs):
            target_spec = MiniGridSpec(target_env, args.max_steps, seed, args.state_encoder)
            config = AdaptationConfig(
                episodes=args.adapt_episodes,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                epsilon=args.adapt_epsilon,
                threshold=args.threshold,
            )
            scratch_report, _scratch_points = evaluate_scratch(target_spec, seed + 1000 + target_idx * 10000, config)
            scratch_row = _report_row(scratch_report, {}, seed)
            rows.append(scratch_row)
            print(_format_progress(scratch_row))

            for idx, artifact in artifacts:
                report, _points = evaluate_transfer(
                    artifact,
                    target_spec,
                    seed + 3000 + target_idx * 10000 + idx * 100,
                    config,
                    scratch_final_score=scratch_report.final_score,
                )
                probe = _target_probe(target_spec, artifact.best_agent(), seed + 5000 + target_idx * 10000 + idx * 100, args.eval_episodes)
                row = _report_row(report, artifact.metadata, seed, probe)
                rows.append(row)
                print(_format_progress(row))
                if args.include_target_reuse:
                    for reuse_idx, reuse_match_mode in enumerate(reuse_match_modes):
                        reuse_config = TargetReuseConfig(
                            probe_episodes=args.reuse_probe_episodes,
                            match_mode=reuse_match_mode,
                            reinforce_passes=args.reuse_reinforce_passes,
                            reward=args.reuse_reward,
                        )
                        reuse_report, _reuse_points, reuse_summary = evaluate_transfer_with_target_reuse(
                            artifact,
                            target_spec,
                            seed + 6000 + target_idx * 10000 + idx * 100 + reuse_idx * 10,
                            config,
                            reuse_config,
                            scratch_final_score=scratch_report.final_score,
                        )
                        reuse_probe = _target_probe(
                            target_spec,
                            artifact.best_agent(),
                            seed + 5000 + target_idx * 10000 + idx * 100,
                            args.eval_episodes,
                        )
                        reuse_row = _report_row(
                            reuse_report,
                            artifact.metadata,
                            seed,
                            reuse_probe,
                            {
                                "target_reuse_match_mode": reuse_match_mode,
                                "target_reuse_item_count": reuse_summary.item_count,
                                "target_reuse_avg_item_score": reuse_summary.avg_item_score,
                                "target_reuse_best_item_score": reuse_summary.best_item_score,
                                "target_reuse_avg_prior_support": reuse_summary.avg_prior_support,
                                "target_reuse_best_prior_support": reuse_summary.best_prior_support,
                                "target_reuse_matched_prior_count": reuse_summary.matched_prior_count,
                                "target_reuse_executed_prior_count": reuse_summary.executed_prior_count,
                                "target_reuse_executed_prior_steps": reuse_summary.executed_prior_steps,
                                "target_reuse_executed_episode_count": reuse_summary.executed_episode_count,
                                "target_reuse_idle_episode_count": reuse_summary.idle_episode_count,
                                "target_reuse_executed_episode_avg_subgoal_score": reuse_summary.executed_episode_avg_subgoal_score,
                                "target_reuse_executed_episode_avg_region_transitions": reuse_summary.executed_episode_avg_region_transitions,
                                "target_reuse_executed_episode_avg_mobility": reuse_summary.executed_episode_avg_mobility,
                                "target_reuse_idle_episode_avg_subgoal_score": reuse_summary.idle_episode_avg_subgoal_score,
                                "target_reuse_idle_episode_avg_region_transitions": reuse_summary.idle_episode_avg_region_transitions,
                                "target_reuse_idle_episode_avg_mobility": reuse_summary.idle_episode_avg_mobility,
                                "target_reuse_probe_success": reuse_summary.probe_success_rate,
                            },
                        )
                        rows.append(reuse_row)
                        print(_format_progress(reuse_row))

    summary_rows = _summary_rows(rows)
    _write_csv(args.output, rows)
    _write_csv(args.summary_output, summary_rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


def _report_row(
    report,
    metadata: dict[str, Any],
    experiment_seed: int,
    target_probe: dict[str, Any] | None = None,
    target_reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    row["target_probe_subgoal_score"] = ""
    row["target_probe_region_transitions"] = ""
    row["target_probe_new_regions"] = ""
    row["target_probe_mobility"] = ""
    row["target_probe_success"] = ""
    row["target_reuse_match_mode"] = ""
    row["target_reuse_item_count"] = ""
    row["target_reuse_avg_item_score"] = ""
    row["target_reuse_best_item_score"] = ""
    row["target_reuse_avg_prior_support"] = ""
    row["target_reuse_best_prior_support"] = ""
    row["target_reuse_matched_prior_count"] = ""
    row["target_reuse_executed_prior_count"] = ""
    row["target_reuse_executed_prior_steps"] = ""
    row["target_reuse_executed_episode_count"] = ""
    row["target_reuse_idle_episode_count"] = ""
    row["target_reuse_executed_episode_avg_subgoal_score"] = ""
    row["target_reuse_executed_episode_avg_region_transitions"] = ""
    row["target_reuse_executed_episode_avg_mobility"] = ""
    row["target_reuse_idle_episode_avg_subgoal_score"] = ""
    row["target_reuse_idle_episode_avg_region_transitions"] = ""
    row["target_reuse_idle_episode_avg_mobility"] = ""
    row["target_reuse_probe_success"] = ""
    if target_probe is not None:
        row.update(target_probe)
    if target_reuse is not None:
        row.update(target_reuse)
    row["artifact_metadata_json"] = json.dumps(metadata, sort_keys=True)
    return row


def _target_probe(target_spec: MiniGridSpec, agent, seed: int, eval_episodes: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(target_spec)
    rollouts = [
        run_minigrid_episode(env, agent, rng, epsilon=0.0, train=False, episode_seed=seed + idx)
        for idx in range(eval_episodes)
    ]
    env.close()
    diagnostics = summarize_rollouts(rollouts, target_spec.max_steps)
    return {
        "target_probe_subgoal_score": diagnostics.avg_subgoal_score,
        "target_probe_region_transitions": diagnostics.avg_region_transitions,
        "target_probe_new_regions": diagnostics.avg_new_regions,
        "target_probe_mobility": diagnostics.avg_mobility,
        "target_probe_success": sum(rollout.success for rollout in rollouts) / max(1, len(rollouts)),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(str(row["target_env"]), str(row["method"])) for row in rows})
    output = []
    for target_env, method in groups:
        method_rows = [row for row in rows if row["target_env"] == target_env and row["method"] == method]
        summary: dict[str, Any] = {
            "target_env": target_env,
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
