from __future__ import annotations
# ruff: noqa: E402

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")


DEFAULT_CANDIDATES = (
    "MiniGrid-FourRooms-v0,"
    "MiniGrid-Dynamic-Obstacles-8x8-v0,"
    "MiniGrid-Dynamic-Obstacles-16x16-v0,"
    "MiniGrid-SimpleCrossingS9N1-v0,"
    "MiniGrid-SimpleCrossingS9N2-v0,"
    "MiniGrid-MultiRoom-N2-S4-v0,"
    "MiniGrid-MultiRoom-N4-S5-v0"
)

DEFAULT_METHODS = (
    "operate_replay_pretrain,"
    "cyclic_motif_fast_replay_pretrain,"
    "cyclic_subgoal_ecology_replay_pretrain"
)

REAL_CONDITIONS = frozenset({"pretrained_agent", "target_reuse"})
CONTROL_CONDITIONS = frozenset({"shuffled_prior_control"})
RANDOM_CONTROL = "random_artifact_control"


@dataclass(frozen=True)
class LadderDecisionConfig:
    min_signal_auc: float = 0.01
    easy_scratch_auc: float = 0.18
    lift_margin: float = 0.03
    control_margin: float = 0.02


@dataclass(frozen=True)
class LadderTargetSummary:
    target_env: str
    tier: str
    reason: str
    scratch_auc: float
    scratch_final: float
    max_auc: float
    max_final: float
    best_real_condition: str
    best_real_source: str
    best_real_runs: int
    best_real_positive_auc_lift_count: int
    best_real_auc: float
    best_real_auc_lift: float
    best_real_final: float
    best_control_condition: str
    best_control_source: str
    best_control_runs: int
    best_control_positive_auc_lift_count: int
    best_control_auc: float
    best_control_final: float
    best_real_region_changes: float
    best_control_region_changes: float
    source_like: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a transfer-task ladder from longitudinal validation outputs.",
    )
    parser.add_argument("--name", default="transfer_ladder_calibration")
    parser.add_argument("--source-env", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--candidate-envs", default=DEFAULT_CANDIDATES)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="geometry")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--pretrain-episodes", type=int, default=24)
    parser.add_argument("--adapt-episodes", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--reuse-probe-episodes", type=int, default=5)
    parser.add_argument("--min-signal-auc", type=float, default=0.01)
    parser.add_argument("--easy-scratch-auc", type=float, default=0.18)
    parser.add_argument("--lift-margin", type=float, default=0.03)
    parser.add_argument("--control-margin", type=float, default=0.02)
    parser.add_argument(
        "--run-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the longitudinal validator before classifying the ladder.",
    )
    parser.add_argument(
        "--summary-input",
        type=Path,
        default=None,
        help="Classify an existing longitudinal summary CSV instead of the default named output.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Raw longitudinal report CSV path.",
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        default=None,
        help="Longitudinal eval-point CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Longitudinal grouped summary CSV path.",
    )
    parser.add_argument(
        "--ladder-output",
        type=Path,
        default=None,
        help="Target-level ladder classification CSV path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Markdown ladder report path.",
    )
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    raw_output = args.raw_output or output_dir / f"{args.name}.csv"
    points_output = args.points_output or output_dir / f"{args.name}_points.csv"
    summary_output = args.summary_output or output_dir / f"{args.name}_summary.csv"
    ladder_output = args.ladder_output or output_dir / f"{args.name}_ladder.csv"
    report_output = args.report_output or output_dir / f"{args.name}_ladder.md"
    summary_input = args.summary_input or summary_output

    if args.run_validation:
        _run_longitudinal_validation(
            source_env=args.source_env,
            candidate_envs=args.candidate_envs,
            seeds=args.seeds,
            max_steps=args.max_steps,
            state_encoder=args.state_encoder,
            methods=args.methods,
            pretrain_episodes=args.pretrain_episodes,
            adapt_episodes=args.adapt_episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            reuse_probe_episodes=args.reuse_probe_episodes,
            raw_output=raw_output,
            points_output=points_output,
            summary_output=summary_output,
        )

    rows = _read_csv(summary_input)
    config = LadderDecisionConfig(
        min_signal_auc=args.min_signal_auc,
        easy_scratch_auc=args.easy_scratch_auc,
        lift_margin=args.lift_margin,
        control_margin=args.control_margin,
    )
    ladder = classify_ladder(rows, args.source_env, config)
    _write_csv(ladder_output, [asdict(item) for item in ladder])
    _write_markdown_report(report_output, ladder, config, summary_input)

    print(f"Wrote {ladder_output}")
    print(f"Wrote {report_output}")


def classify_ladder(
    summary_rows: list[dict[str, str]],
    source_env: str,
    config: LadderDecisionConfig,
) -> list[LadderTargetSummary]:
    targets = sorted({row["target_env"] for row in summary_rows})
    return [
        classify_target(
            [row for row in summary_rows if row["target_env"] == target],
            source_env,
            config,
        )
        for target in targets
    ]


def classify_target(
    rows: list[dict[str, str]],
    source_env: str,
    config: LadderDecisionConfig,
) -> LadderTargetSummary:
    if not rows:
        msg = "cannot classify an empty target row set"
        raise ValueError(msg)

    target_env = rows[0]["target_env"]
    scratch = _find_scratch(rows)
    real_rows = [row for row in rows if _is_real_condition(row)]
    control_rows = [row for row in rows if _is_control_condition(row)]
    best_real = _best_row(real_rows, "adaptation_auc_mean")
    best_control = _best_row(control_rows, "adaptation_auc_mean")
    best_real_region = _best_row(real_rows, "target_reuse_composition_region_change_count_mean")
    best_control_region = _best_row(control_rows, "target_reuse_composition_region_change_count_mean")

    scratch_auc = _float(scratch.get("adaptation_auc_mean", ""))
    scratch_final = _float(scratch.get("final_score_mean", ""))
    max_auc = max((_float(row.get("adaptation_auc_mean", "")) for row in rows), default=0.0)
    max_final = max((_float(row.get("final_score_mean", "")) for row in rows), default=0.0)
    best_real_auc = _float(best_real.get("adaptation_auc_mean", ""))
    best_real_lift = _float(best_real.get("adaptation_auc_lift_mean", ""))
    best_control_auc = _float(best_control.get("adaptation_auc_mean", ""))
    best_real_runs = _int(best_real.get("runs", ""))
    best_real_positive_auc_lift_count = _int(best_real.get("positive_auc_lift_count", ""))
    best_control_runs = _int(best_control.get("runs", ""))
    best_control_positive_auc_lift_count = _int(best_control.get("positive_auc_lift_count", ""))

    tier, reason = _tier_reason(
        target_env=target_env,
        source_env=source_env,
        scratch_auc=scratch_auc,
        max_auc=max_auc,
        max_final=max_final,
        best_real_auc=best_real_auc,
        best_real_lift=best_real_lift,
        best_real_runs=best_real_runs,
        best_real_positive_auc_lift_count=best_real_positive_auc_lift_count,
        best_control_auc=best_control_auc,
        config=config,
    )

    return LadderTargetSummary(
        target_env=target_env,
        tier=tier,
        reason=reason,
        scratch_auc=scratch_auc,
        scratch_final=scratch_final,
        max_auc=max_auc,
        max_final=max_final,
        best_real_condition=best_real.get("condition", ""),
        best_real_source=best_real.get("source_method", ""),
        best_real_runs=best_real_runs,
        best_real_positive_auc_lift_count=best_real_positive_auc_lift_count,
        best_real_auc=best_real_auc,
        best_real_auc_lift=best_real_lift,
        best_real_final=_float(best_real.get("final_score_mean", "")),
        best_control_condition=best_control.get("condition", ""),
        best_control_source=best_control.get("source_method", ""),
        best_control_runs=best_control_runs,
        best_control_positive_auc_lift_count=best_control_positive_auc_lift_count,
        best_control_auc=best_control_auc,
        best_control_final=_float(best_control.get("final_score_mean", "")),
        best_real_region_changes=_float(
            best_real_region.get("target_reuse_composition_region_change_count_mean", "")
        ),
        best_control_region_changes=_float(
            best_control_region.get("target_reuse_composition_region_change_count_mean", "")
        ),
        source_like=target_env == source_env,
    )


def _tier_reason(
    *,
    target_env: str,
    source_env: str,
    scratch_auc: float,
    max_auc: float,
    max_final: float,
    best_real_auc: float,
    best_real_lift: float,
    best_real_runs: int,
    best_real_positive_auc_lift_count: int,
    best_control_auc: float,
    config: LadderDecisionConfig,
) -> tuple[str, str]:
    if target_env == source_env:
        return "source_like_regression", "same environment as source; use for regression, not transfer claims"
    if max_auc <= config.min_signal_auc and max_final <= 0.0:
        return "all_zero_hard", "no condition produced task-level signal under this budget"
    if scratch_auc >= config.easy_scratch_auc and best_real_lift <= config.lift_margin:
        return "scratch_dominated", "scratch already has strong AUC and real pretraining does not clear lift margin"
    if best_real_auc > config.min_signal_auc and best_control_auc + config.control_margin >= best_real_auc:
        return "control_confounded", "negative controls match or exceed the best real pretraining signal"
    if best_real_lift > config.lift_margin and best_real_auc > best_control_auc + config.control_margin:
        if best_real_runs > 1 and best_real_positive_auc_lift_count < best_real_runs:
            return "unstable_transfer_signal", "mean clears margins but positive lift is not stable across seeds"
        return "candidate_transfer_signal", "real pretraining clears scratch lift and negative-control margins"
    return "measurable_candidate", "task has nonzero signal but no clean transfer claim yet"


def _run_longitudinal_validation(
    *,
    source_env: str,
    candidate_envs: str,
    seeds: str,
    max_steps: int,
    state_encoder: str,
    methods: str,
    pretrain_episodes: int,
    adapt_episodes: int,
    eval_every: int,
    eval_episodes: int,
    reuse_probe_episodes: int,
    raw_output: Path,
    points_output: Path,
    summary_output: Path,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_pretraining_benefit_longitudinal.py",
        "--source-env",
        source_env,
        "--target-envs",
        candidate_envs,
        "--seeds",
        seeds,
        "--max-steps",
        str(max_steps),
        "--state-encoder",
        state_encoder,
        "--methods",
        methods,
        "--pretrain-episodes",
        str(pretrain_episodes),
        "--adapt-episodes",
        str(adapt_episodes),
        "--eval-every",
        str(eval_every),
        "--eval-episodes",
        str(eval_episodes),
        "--reuse-probe-episodes",
        str(reuse_probe_episodes),
        "--output",
        str(raw_output),
        "--points-output",
        str(points_output),
        "--summary-output",
        str(summary_output),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _find_scratch(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("condition") == "scratch" and row.get("source_method") == "scratch":
            return row
    return {}


def _is_real_condition(row: dict[str, str]) -> bool:
    return row.get("condition") in REAL_CONDITIONS and row.get("source_method") != RANDOM_CONTROL


def _is_control_condition(row: dict[str, str]) -> bool:
    return row.get("condition") in CONTROL_CONDITIONS or row.get("source_method") == RANDOM_CONTROL


def _best_row(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: _float(row.get(metric, "")))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(
    path: Path,
    ladder: list[LadderTargetSummary],
    config: LadderDecisionConfig,
    summary_input: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transfer Ladder Calibration",
        "",
        "Source summary:",
        "",
        f"```text\n{summary_input}\n```",
        "",
        "Decision thresholds:",
        "",
        "```text",
        f"min_signal_auc = {config.min_signal_auc}",
        f"easy_scratch_auc = {config.easy_scratch_auc}",
        f"lift_margin = {config.lift_margin}",
        f"control_margin = {config.control_margin}",
        "```",
        "",
        "| Target | Tier | Scratch AUC | Best Real AUC | Best Control AUC | Reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in ladder:
        lines.append(
            "| "
            f"`{item.target_env}` | `{item.tier}` | {item.scratch_auc:.4f} | "
            f"{item.best_real_auc:.4f} | {item.best_control_auc:.4f} | {item.reason} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _float(value: object) -> float:
    if not isinstance(value, int | float | str):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    if not isinstance(value, int | float | str):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
