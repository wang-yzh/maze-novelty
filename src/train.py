from __future__ import annotations
# ruff: noqa: E402

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from agents import QAgent, run_episode
from config import load_config
from env import GridWorld, MazeSpec, generate_mazes
from evolution import (
    evaluate_agent,
    evolve_population,
    make_population,
    train_population_novelty,
    train_population_task,
    train_population_task_with_replay,
    exploitation_score,
)
from novelty import NoveltyArchive
from replay import SuccessReplayBank
from visualize import plot_paths, plot_summary, write_metrics_csv


OUTPUTS = ROOT / "outputs"
METHOD_SEED_OFFSETS = {
    "q_learning": 101,
    "novelty_q": 211,
    "genetic_q": 307,
    "cyclic_novelty": 401,
    "cyclic_replay": 503,
    "cyclic_restart": 607,
    "cyclic_replay_restart": 709,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GridWorld novelty experiments.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--seed", type=int, help="Override the seed from the config file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--methods", help="Comma-separated method names to run.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation.")
    parser.add_argument("--dry-run", action="store_true", help="Load config and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_plots:
        config["make_plots"] = False
    if args.dry_run:
        print(f"Loaded config: {config}")
        return

    spec = MazeSpec(
        size=config["size"],
        obstacle_prob=config["obstacle_prob"],
        max_steps=config["size"] ** 2,
    )
    train_grids = generate_mazes(config["train_mazes"], config["seed"], spec)
    test_grids = generate_mazes(config["test_mazes"], config["seed"] + 1000, spec)
    n_states = GridWorld(train_grids[0]).n_states

    rows: list[dict] = []
    representative_paths: dict[str, list[tuple[int, int]]] = {}

    methods = {
        "q_learning": run_q_learning,
        "novelty_q": run_novelty_q,
        "genetic_q": run_genetic_q,
        "cyclic_novelty": run_cyclic_novelty,
        "cyclic_replay": run_cyclic_replay,
        "cyclic_restart": run_cyclic_restart,
        "cyclic_replay_restart": run_cyclic_replay_restart,
    }
    if args.methods:
        requested = [method.strip() for method in args.methods.split(",") if method.strip()]
        unknown = sorted(set(requested) - set(methods))
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        methods = {name: methods[name] for name in requested}
    for name, runner in methods.items():
        print(f"Running {name}...")
        method_rng = np.random.default_rng(config["seed"] + METHOD_SEED_OFFSETS[name])
        method_rows, path = runner(config, train_grids, test_grids, method_rng, n_states)
        rows.extend(method_rows)
        representative_paths[name] = path

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    if config["make_plots"]:
        plot_summary(output_dir / "summary.png", rows)
        plot_paths(output_dir / "paths.png", test_grids[0], representative_paths)
    print(f"Done. Outputs written to {output_dir}")


def run_q_learning(config, train_grids, test_grids, rng, n_states):
    agent = QAgent(n_states, 4, rng)
    archive = NoveltyArchive(config["size"])
    rows = []
    best_path = []
    for gen in range(config["generations"] + 1):
        epsilon = max(0.05, 0.35 * (1.0 - gen / config["generations"]))
        for _ in range(config["population"] * config["episodes_per_agent"]):
            grid = train_grids[int(rng.integers(len(train_grids)))]
            rollout = run_episode(GridWorld(grid), agent, epsilon=epsilon, train=True)
            archive.add(rollout.positions, rollout.success)
        if gen % config["eval_every"] == 0 or gen == config["generations"]:
            result = evaluate_agent(agent, test_grids, config["size"] ** 2)
            rows.append(_row("q_learning", gen, result, archive, "task"))
            best_path = result.best_rollout.positions
    return rows, best_path


def run_novelty_q(config, train_grids, test_grids, rng, n_states):
    agent = QAgent(n_states, 4, rng)
    archive = NoveltyArchive(config["size"])
    rows = []
    best_path = []
    for gen in range(config["generations"] + 1):
        epsilon = max(0.06, 0.40 * (1.0 - gen / config["generations"]))
        bonus_weight = max(0.02, 0.15 * (1.0 - gen / config["generations"]))
        for _ in range(config["population"] * config["episodes_per_agent"]):
            grid = train_grids[int(rng.integers(len(train_grids)))]
            rollout = run_episode(
                GridWorld(grid),
                agent,
                epsilon=epsilon,
                train=True,
                novelty_bonus=lambda _s, pos: bonus_weight * archive.position_bonus(pos),
            )
            archive.add(rollout.positions, rollout.success)
        if gen % config["eval_every"] == 0 or gen == config["generations"]:
            result = evaluate_agent(agent, test_grids, config["size"] ** 2)
            rows.append(_row("novelty_q", gen, result, archive, "novelty"))
            best_path = result.best_rollout.positions
    return rows, best_path


def run_genetic_q(config, train_grids, test_grids, rng, n_states):
    population = make_population(config["population"], n_states, 4, rng)
    archive = NoveltyArchive(config["size"])
    rows = []
    best_path = []
    for gen in range(config["generations"] + 1):
        epsilon = max(0.05, 0.35 * (1.0 - gen / config["generations"]))
        train_population_task(
            population,
            train_grids,
            rng,
            config["episodes_per_agent"],
            config["size"] ** 2,
            epsilon,
        )
        scores = []
        for agent in population:
            result = evaluate_agent(agent, train_grids[:6], config["size"] ** 2)
            scores.append(result.score)
            archive.add(result.best_rollout.positions, result.best_rollout.success)
        population = evolve_population(population, scores, rng)
        if gen % config["eval_every"] == 0 or gen == config["generations"]:
            best = _best_agent(population, test_grids, config["size"] ** 2)
            rows.append(_row("genetic_q", gen, best, archive, "exploit"))
            best_path = best.best_rollout.positions
    return rows, best_path


def run_cyclic_novelty(config, train_grids, test_grids, rng, n_states):
    return _run_cyclic_variant(
        "cyclic_novelty",
        config,
        train_grids,
        test_grids,
        rng,
        n_states,
        use_replay=False,
        use_restart=False,
    )


def run_cyclic_replay(config, train_grids, test_grids, rng, n_states):
    return _run_cyclic_variant(
        "cyclic_replay",
        config,
        train_grids,
        test_grids,
        rng,
        n_states,
        use_replay=True,
        use_restart=False,
    )


def run_cyclic_restart(config, train_grids, test_grids, rng, n_states):
    return _run_cyclic_variant(
        "cyclic_restart",
        config,
        train_grids,
        test_grids,
        rng,
        n_states,
        use_replay=False,
        use_restart=True,
    )


def run_cyclic_replay_restart(config, train_grids, test_grids, rng, n_states):
    return _run_cyclic_variant(
        "cyclic_replay_restart",
        config,
        train_grids,
        test_grids,
        rng,
        n_states,
        use_replay=True,
        use_restart=True,
    )


def _run_cyclic_variant(
    method,
    config,
    train_grids,
    test_grids,
    rng,
    n_states,
    use_replay,
    use_restart,
):
    population = make_population(config["population"], n_states, 4, rng)
    hall_of_fame = [agent.clone() for agent in population[:2]]
    replay_bank = SuccessReplayBank() if use_replay else None
    archive = NoveltyArchive(config["size"])
    rows = []
    best_path = []
    phase = "novelty"
    no_positive = 0
    novelty_age = 0
    exploit_age = 0
    novelty_max_age = 10
    exploit_min_age = 8
    for gen in range(config["generations"] + 1):
        if phase == "novelty":
            epsilon = 0.35
            bonus_weight = 0.04
            novelty_scores = train_population_novelty(
                population,
                train_grids,
                archive,
                rng,
                config["episodes_per_agent"],
                config["size"] ** 2,
                epsilon,
                bonus_weight,
            )
            time_penalty = novelty_age / max(1, config["novelty_patience"] * 2)
            gated_scores = [score - time_penalty for score in novelty_scores]
            if max(gated_scores) <= 0.0:
                no_positive += 1
            else:
                no_positive = 0
            population = evolve_population(
                population,
                _hidden_scores(population, train_grids, archive, config["size"] ** 2),
                rng,
                mutation_scale=0.10,
                mutation_rate=0.10,
            )
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
            novelty_age += 1
            if no_positive >= config["novelty_patience"] or novelty_age >= novelty_max_age:
                phase = "exploit"
                no_positive = 0
                novelty_age = 0
                exploit_age = 0
        else:
            epsilon = max(0.05, 0.28 * (1.0 - gen / config["generations"]))
            train_population_task_with_replay(
                population,
                train_grids,
                rng,
                config["episodes_per_agent"] + 6,
                config["size"] ** 2,
                epsilon,
                replay_bank,
            )
            train_results = [
                evaluate_agent(agent, train_grids[:8], config["size"] ** 2) for agent in population
            ]
            if replay_bank is not None:
                for result in train_results:
                    replay_bank.add(result.best_rollout)
            exploit_scores = [
                exploitation_score(result, config["size"] ** 2) for result in train_results
            ]
            hall_of_fame = _update_hall_of_fame(
                hall_of_fame,
                population,
                train_results,
                train_grids[:8],
                config["size"] ** 2,
                limit=max(2, config["population"] // 4),
            )
            population = evolve_population(
                population,
                exploit_scores,
                rng,
                mutation_scale=0.025,
                mutation_rate=0.04,
            )
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
            if replay_bank is not None:
                for agent in population[: len(hall_of_fame)]:
                    replay_bank.reinforce(agent, rng, passes=3)
            exploit_age += 1
            best_train_success = max(result.success_rate for result in train_results)
            should_restart = use_restart and exploit_age >= exploit_min_age and best_train_success < 0.25
            if should_restart:
                population = _restart_population(
                    hall_of_fame,
                    replay_bank,
                    config["population"],
                    n_states,
                    rng,
                )
                phase = "novelty"
                exploit_age = 0
            elif exploit_age >= exploit_min_age:
                phase = "novelty"
                exploit_age = 0

        if gen % config["eval_every"] == 0 or gen == config["generations"]:
            best = _best_agent(population + hall_of_fame, test_grids, config["size"] ** 2)
            rows.append(_row(method, gen, best, archive, phase))
            best_path = best.best_rollout.positions
    return rows, best_path


def _hidden_scores(population, train_grids, archive, max_steps):
    scores = []
    for agent in population:
        result = evaluate_agent(agent, train_grids[:6], max_steps)
        novelty = archive.trajectory_novelty(result.best_rollout.positions)
        speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
        hidden = (
            0.30 * result.success_rate
            + 0.25 * speed
            + 0.25 * novelty
            + 0.20 * result.stability
        )
        scores.append(hidden)
    return scores


def _best_agent(population, grids, max_steps):
    results = [evaluate_agent(agent, grids, max_steps) for agent in population]
    return max(results, key=lambda r: r.score)


def _restart_population(hall_of_fame, replay_bank, population_size, n_states, rng):
    next_population = []
    elite_count = min(len(hall_of_fame), max(2, population_size // 5))
    for agent in hall_of_fame[:elite_count]:
        next_population.append(agent.clone())

    while len(next_population) < population_size:
        if hall_of_fame and len(next_population) < int(population_size * 0.5):
            base = hall_of_fame[int(rng.integers(len(hall_of_fame)))]
            child = replay_bank.seed_agent(base, rng) if replay_bank is not None else base.clone()
            child.mutate(scale=0.16, rate=0.14)
        else:
            child = make_population(1, n_states, 4, rng)[0]
            if replay_bank is not None and len(replay_bank) > 0:
                child = replay_bank.seed_agent(child, rng)
                child.mutate(scale=0.10, rate=0.08)
        next_population.append(child)
    return next_population


def _update_hall_of_fame(hall_of_fame, population, results, eval_grids, max_steps, limit):
    candidates = [
        (agent, evaluate_agent(agent, eval_grids, max_steps)) for agent in hall_of_fame
    ]
    candidates.extend(zip(population, results, strict=True))
    ranked = sorted(
        candidates,
        key=lambda item: exploitation_score(item[1], max_steps),
        reverse=True,
    )
    return [agent.clone() for agent, _ in ranked[:limit]]


def _row(method, generation, result, archive, phase):
    return {
        "method": method,
        "generation": generation,
        "phase": phase,
        "test_success": round(result.success_rate, 4),
        "test_avg_steps": round(result.avg_steps, 4),
        "test_stability": round(result.stability, 4),
        "test_score": round(result.score, 4),
        "archive_coverage": round(archive.coverage(), 4),
        "archive_unique_ratio": round(archive.unique_ratio(), 4),
        "success_path_diversity": round(archive.success_diversity(), 4),
    }


if __name__ == "__main__":
    main()
