from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable MiniGrid benchmark jobs.")
    parser.add_argument("--env-id", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--name", default="fourrooms")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--episodes-per-agent", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="compact")
    parser.add_argument("--method-time-limit-seconds", type=float, default=0.0)
    parser.add_argument(
        "--methods",
        default="q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_operate_replay,go_explore_lite,map_elites_lite",
    )
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    run_dirs = []
    for seed in seeds:
        output_dir = ROOT / "outputs" / f"minigrid_{args.name}_seed_{seed}"
        metrics = output_dir / "metrics.csv"
        run_dirs.append(output_dir)
        if metrics.exists():
            print(f"Skipping seed {seed}: {metrics} exists")
            continue
        cmd = [
            sys.executable,
            "src/minigrid_train.py",
            "--env-id",
            args.env_id,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
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
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    summary_path = ROOT / "outputs" / f"minigrid_{args.name}_summary.csv"
    summarize_cmd = [
        sys.executable,
        "scripts/summarize_runs.py",
        *[str(path) for path in run_dirs],
        "--csv-out",
        str(summary_path),
    ]
    print("Summarizing:", " ".join(summarize_cmd))
    subprocess.run(summarize_cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
