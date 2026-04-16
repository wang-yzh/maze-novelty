from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small MiniGrid task ladder.")
    parser.add_argument(
        "--tasks",
        default="MiniGrid-FourRooms-v0,MiniGrid-MultiRoom-N4-S5-v0,MiniGrid-Dynamic-Obstacles-8x8-v0",
    )
    parser.add_argument("--name", default="task_suite")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--episodes-per-agent", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="geometry")
    parser.add_argument("--method-time-limit-seconds", type=float, default=100.0)
    parser.add_argument(
        "--methods",
        default="cyclic_operate_replay,cyclic_motif_fast_replay,cyclic_ecology,cyclic_motif_bootstrap_replay",
    )
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    summary_paths = []
    for task in tasks:
        task_slug = _slug(task)
        benchmark_name = f"{args.name}_{task_slug}"
        cmd = [
            sys.executable,
            "scripts/run_minigrid_benchmark.py",
            "--env-id",
            task,
            "--name",
            benchmark_name,
            "--seeds",
            args.seeds,
            "--generations",
            str(args.generations),
            "--population",
            str(args.population),
            "--episodes-per-agent",
            str(args.episodes_per_agent),
            "--eval-episodes",
            str(args.eval_episodes),
            "--eval-every",
            str(args.eval_every),
            "--max-steps",
            str(args.max_steps),
            "--state-encoder",
            args.state_encoder,
            "--method-time-limit-seconds",
            str(args.method_time_limit_seconds),
            "--methods",
            args.methods,
        ]
        print("Running task:", task)
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        summary_paths.append(ROOT / "outputs" / f"minigrid_{benchmark_name}_summary.csv")

    print("\nTask summaries:")
    for path in summary_paths:
        print(path)


def _slug(value: str) -> str:
    return (
        value.replace("MiniGrid-", "")
        .replace("-v0", "")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


if __name__ == "__main__":
    main()
