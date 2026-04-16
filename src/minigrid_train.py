from __future__ import annotations
# ruff: noqa: E402

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from agents import QAgent, Rollout
from core.budget import MethodBudget
from core.recording import annotate_method_rows
from ecology import (
    MotifBank,
    NicheArchive,
    behavior_descriptor,
    most_mobile_window,
    motif_selection_score,
    reinforce_action_trace,
    resource_selection_score,
    speciated_selection_score,
    stress_score,
)
from evolution import EvalResult, evolve_population, exploitation_score
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from novelty import NoveltyArchive
from replay import SuccessReplayBank
from visualize import write_metrics_csv


METHOD_SEED_OFFSETS = {
    "q_learning": 101,
    "q_learning_strong": 151,
    "genetic_q": 307,
    "genetic_q_strong": 353,
    "cyclic_novelty": 401,
    "cyclic_replay": 503,
    "cyclic_three_phase": 557,
    "cyclic_operate_replay": 601,
    "cyclic_ecology": 653,
    "cyclic_ecology_no_motif": 661,
    "cyclic_ecology_no_stress": 673,
    "cyclic_ecology_niche_only": 677,
    "cyclic_ecology_no_bottleneck": 683,
    "cyclic_speciated_stress_replay": 691,
    "cyclic_motif_oriented_radiation": 697,
    "cyclic_motif_fast_replay": 698,
    "cyclic_resource_ecology_replay": 699,
    "go_explore_lite": 701,
    "map_elites_lite": 809,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tabular methods on MiniGrid.")
    parser.add_argument("--env-id", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "minigrid")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--episodes-per-agent", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="compact")
    parser.add_argument("--methods", default="q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_operate_replay")
    parser.add_argument("--method-time-limit-seconds", type=float, default=0.0)
    args = parser.parse_args()

    methods = {
        "q_learning": run_q_learning,
        "q_learning_strong": run_q_learning_strong,
        "genetic_q": run_genetic_q,
        "genetic_q_strong": run_genetic_q_strong,
        "cyclic_novelty": run_cyclic_novelty,
        "cyclic_replay": run_cyclic_replay,
        "cyclic_three_phase": run_cyclic_three_phase,
        "cyclic_operate_replay": run_cyclic_operate_replay,
        "cyclic_ecology": run_cyclic_ecology,
        "cyclic_ecology_no_motif": run_cyclic_ecology_no_motif,
        "cyclic_ecology_no_stress": run_cyclic_ecology_no_stress,
        "cyclic_ecology_niche_only": run_cyclic_ecology_niche_only,
        "cyclic_ecology_no_bottleneck": run_cyclic_ecology_no_bottleneck,
        "cyclic_speciated_stress_replay": run_cyclic_speciated_stress_replay,
        "cyclic_motif_oriented_radiation": run_cyclic_motif_oriented_radiation,
        "cyclic_motif_fast_replay": run_cyclic_motif_fast_replay,
        "cyclic_resource_ecology_replay": run_cyclic_resource_ecology_replay,
        "go_explore_lite": run_go_explore_lite,
        "map_elites_lite": run_map_elites_lite,
    }
    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    rows = []
    for method in requested:
        print(f"Running {method}...")
        rng = np.random.default_rng(args.seed + METHOD_SEED_OFFSETS[method])
        args.method_budget = MethodBudget.from_seconds(args.method_time_limit_seconds).start()
        method_rows = methods[method](args, rng)
        method_rows, runtime_seconds = annotate_method_rows(method_rows, args.method_budget)
        for row in method_rows:
            row["state_encoder"] = args.state_encoder
        rows.extend(method_rows)
        print(f"{method} runtime_seconds={runtime_seconds:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(args.output_dir / "metrics.csv", rows)
    print(f"Done. Outputs written to {args.output_dir}")


def _time_expired(args) -> bool:
    budget = getattr(args, "method_budget", None)
    return bool(budget is not None and budget.expired())


def _advance_phase(schedule, phase_index: int, phase_age: int, phase_len: int) -> tuple[int, int]:
    phase_age += 1
    if phase_age >= phase_len:
        phase_index = (phase_index + 1) % len(schedule)
        phase_age = 0
    return phase_index, phase_age


def run_q_learning(args, rng):
    return _run_q_learning(args, rng, "q_learning", episode_multiplier=1)


def run_q_learning_strong(args, rng):
    return _run_q_learning(args, rng, "q_learning_strong", episode_multiplier=4)


def _run_q_learning(args, rng, method: str, episode_multiplier: int):
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    agent = QAgent(probe.n_states, probe.n_actions, rng)
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    rows = []
    env = MiniGridTabularEnv(spec)
    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        epsilon = max(0.05, 0.45 * (1.0 - gen / args.generations))
        episode_count = args.population * args.episodes_per_agent * episode_multiplier
        for episode in range(episode_count):
            rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, episode)
            archive.add(rollout.positions, rollout.success)
        if gen % args.eval_every == 0 or gen == args.generations:
            result = _evaluate_minigrid(args, agent, rng)
            rows.append(_row(method, gen, result, archive, "task"))
    env.close()
    return rows


def run_genetic_q(args, rng):
    return _run_genetic_q(args, rng, "genetic_q", episode_multiplier=1, eval_count=6)


def run_genetic_q_strong(args, rng):
    return _run_genetic_q(args, rng, "genetic_q_strong", episode_multiplier=2, eval_count=10)


def _run_genetic_q(args, rng, method: str, episode_multiplier: int, eval_count: int):
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    rows = []
    env = MiniGridTabularEnv(spec)
    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        epsilon = max(0.05, 0.35 * (1.0 - gen / args.generations))
        for idx, agent in enumerate(population):
            for episode in range(args.episodes_per_agent * episode_multiplier):
                rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                archive.add(rollout.positions, rollout.success)
        results = [_evaluate_minigrid(args, agent, rng, eval_episodes=eval_count) for agent in population]
        scores = [result.score for result in results]
        population = evolve_population(
            population,
            scores,
            rng,
            elite_frac=0.30,
            mutation_scale=0.05 if episode_multiplier > 1 else 0.07,
            mutation_rate=0.05 if episode_multiplier > 1 else 0.07,
        )
        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population), key=lambda r: r.score)
            rows.append(_row(method, gen, best, archive, "exploit"))
    env.close()
    return rows


def run_cyclic_novelty(args, rng):
    return _run_cyclic(args, rng, method="cyclic_novelty", use_replay=False)


def run_cyclic_replay(args, rng):
    return _run_cyclic(args, rng, method="cyclic_replay", use_replay=True)


def run_cyclic_three_phase(args, rng):
    return _run_cyclic_three_phase(args, rng)


def run_cyclic_operate_replay(args, rng):
    return _run_cyclic_operate_replay(args, rng)


def run_cyclic_ecology(args, rng):
    return _run_cyclic_ecology(args, rng, method="cyclic_ecology")


def run_cyclic_ecology_no_motif(args, rng):
    return _run_cyclic_ecology(args, rng, method="cyclic_ecology_no_motif", use_motif=False)


def run_cyclic_ecology_no_stress(args, rng):
    return _run_cyclic_ecology(args, rng, method="cyclic_ecology_no_stress", use_stress=False)


def run_cyclic_ecology_niche_only(args, rng):
    return _run_cyclic_ecology(
        args,
        rng,
        method="cyclic_ecology_niche_only",
        use_motif=False,
        use_stress=False,
        use_bottleneck=False,
        niche_only=True,
    )


def run_cyclic_ecology_no_bottleneck(args, rng):
    return _run_cyclic_ecology(args, rng, method="cyclic_ecology_no_bottleneck", use_bottleneck=False)


def run_cyclic_speciated_stress_replay(args, rng):
    return _run_cyclic_speciated_stress_replay(args, rng)


def run_cyclic_motif_oriented_radiation(args, rng):
    return _run_cyclic_motif_oriented_radiation(args, rng)


def run_cyclic_motif_fast_replay(args, rng):
    return _run_cyclic_motif_fast_replay(args, rng)


def run_cyclic_resource_ecology_replay(args, rng):
    return _run_cyclic_resource_ecology_replay(args, rng)


def run_go_explore_lite(args, rng):
    return _run_go_explore_lite(args, rng)


def run_map_elites_lite(args, rng):
    return _run_map_elites_lite(args, rng)


def _run_cyclic(args, rng, method: str, use_replay: bool):
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank() if use_replay else None
    rows = []
    phase = "novelty"
    phase_age = 0
    env = MiniGridTabularEnv(spec)
    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        if phase == "novelty":
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.35,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.03,
                    )
                    archive.add(rollout.positions, rollout.success)
                    if replay_bank is not None:
                        replay_bank.add(rollout)
            scores = [_hidden_score(args, agent, archive, rng) for agent in population]
            population = evolve_population(population, scores, rng, mutation_scale=0.10, mutation_rate=0.10)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
            phase_age += 1
            if phase_age >= 8:
                phase = "exploit"
                phase_age = 0
        else:
            epsilon = max(0.05, 0.25 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 4):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    if replay_bank is not None:
                        replay_bank.add(rollout)
                if replay_bank is not None:
                    replay_bank.reinforce(agent, rng, passes=2)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
            if replay_bank is not None:
                for agent in population[: len(hall_of_fame)]:
                    replay_bank.reinforce(agent, rng, passes=3)
            phase_age += 1
            if phase_age >= 8:
                phase = "novelty"
                phase_age = 0

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(_row(method, gen, best, archive, phase))
    env.close()
    return rows


def _run_cyclic_operate_replay(args, rng):
    method = "cyclic_operate_replay"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank(max_items=160)
    rows = []
    schedule = ("explore_a", "operate_a", "explore_b", "operate_b", "replay")
    phase_index = 0
    phase_age = 0
    phase_len = 4
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = schedule[phase_index]
        if phase.startswith("explore"):
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.40,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.035,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
            scores = [_hidden_score(args, agent, archive, rng) for agent in population]
            population = evolve_population(population, scores, rng, mutation_scale=0.11, mutation_rate=0.11)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
        elif phase.startswith("operate"):
            epsilon = max(0.05, 0.24 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 4):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
        else:
            for agent in population:
                replay_bank.reinforce(agent, rng, passes=8, reward=0.10)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=8) for agent in population]
            scores = [0.70 * result.success_rate + 0.20 * result.score + 0.10 * (1.0 - result.avg_steps / args.max_steps) for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.35, mutation_scale=0.02, mutation_rate=0.03)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_age += 1
        if phase_age >= phase_len:
            phase_index = (phase_index + 1) % len(schedule)
            phase_age = 0

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(_row(method, gen, best, archive, schedule[phase_index]))
    env.close()
    return rows


def _run_cyclic_ecology(
    args,
    rng,
    method: str,
    use_motif: bool = True,
    use_stress: bool = True,
    use_bottleneck: bool = True,
    niche_only: bool = False,
):
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank(max_items=180)
    niche_archive = NicheArchive()
    motif_bank = MotifBank()
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    stress_survivors = [agent.clone() for agent in hall_of_fame]
    rows = []
    schedule = ("radiation", "niche", "reradiation", "niche") if niche_only else ("radiation", "niche", "stress", "bottleneck", "reradiation", "consolidation")
    phase_index = 0
    phase_age = 0
    phase_len = 4
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = schedule[phase_index]

        if phase == "radiation":
            scores = []
            for idx, agent in enumerate(population):
                rollout_scores = []
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.46,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.045,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    if use_motif:
                        motif_bank.add_success(rollout)
                    novelty = archive.trajectory_novelty(rollout.positions)
                    coverage = len(set(rollout.positions)) / args.max_steps
                    speed = 1.0 - min(rollout.steps, args.max_steps) / args.max_steps
                    rollout_scores.append(0.55 * novelty + 0.25 * coverage + 0.10 * float(rollout.success) + 0.10 * speed)
                    niche_archive.add(agent, rollout, rollout_scores[-1], args.max_steps)
                scores.append(float(np.mean(rollout_scores)))
            population = evolve_population(population, scores, rng, mutation_scale=0.12, mutation_rate=0.12)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        elif phase == "niche":
            scores = []
            for agent in population:
                result = _evaluate_minigrid(args, agent, rng, eval_episodes=4)
                niche_score = result.score + 0.08 * result.success_rate
                niche_archive.add(agent, result.best_rollout, niche_score, args.max_steps)
                scores.append(niche_score + 0.001 * len(niche_archive))
            niche_elites = niche_archive.best(max(2, args.population // 4))
            population = evolve_population(population, scores, rng, elite_frac=0.30, mutation_scale=0.06, mutation_rate=0.07)
            population[: len(niche_elites)] = niche_elites[: len(population)]

        elif phase == "stress":
            epsilon = max(0.05, 0.22 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 2):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    if use_motif:
                        motif_bank.add_success(rollout)
            if use_stress:
                results = [_evaluate_minigrid_stress(args, agent, rng, gen, eval_episodes=4) for agent in population]
                scores = [_stress_score(result, args.max_steps) for result in results]
            else:
                results = [_evaluate_minigrid(args, agent, rng, eval_episodes=4) for agent in population]
                scores = [exploitation_score(result, args.max_steps) for result in results]
            order = np.argsort(scores)[::-1]
            stress_survivors = [population[int(index)].clone() for index in order[: max(2, args.population // 4)]]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.035, mutation_rate=0.05)
            population[: len(stress_survivors)] = [agent.clone() for agent in stress_survivors]

        elif phase == "bottleneck":
            if use_bottleneck:
                population = _ecology_bottleneck(args, population, stress_survivors, niche_archive, hall_of_fame, rng, probe.n_states)
            else:
                candidates = stress_survivors + niche_archive.best(max(2, args.population // 4)) + hall_of_fame + population
                scores = [_evaluate_minigrid(args, agent, rng, eval_episodes=4).score for agent in candidates]
                population = evolve_population(candidates[: args.population], scores[: args.population], rng, mutation_scale=0.04, mutation_rate=0.05)

        elif phase == "reradiation":
            scores = []
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.38,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.035,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    if use_motif:
                        motif_bank.add_success(rollout)
                result = _evaluate_minigrid(args, agent, rng, eval_episodes=4)
                novelty = archive.trajectory_novelty(result.best_rollout.positions)
                scores.append(0.35 * result.score + 0.35 * novelty + 0.30 * result.success_rate)
            population = evolve_population(population, scores, rng, mutation_scale=0.09, mutation_rate=0.10)

        else:
            for agent in population:
                replay_bank.reinforce(agent, rng, passes=4, reward=0.09)
                if use_motif:
                    motif_bank.reinforce(agent, rng, passes=3, reward=0.07)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) + 0.05 * result.success_rate for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.35, mutation_scale=0.025, mutation_rate=0.035)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_age += 1
        if phase_age >= phase_len:
            phase_index = (phase_index + 1) % len(schedule)
            phase_age = 0

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(_row(method, gen, best, archive, schedule[phase_index]))

    env.close()
    return rows


def _run_cyclic_speciated_stress_replay(args, rng):
    method = "cyclic_speciated_stress_replay"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank(max_items=180)
    niche_archive = NicheArchive(max_per_cell=2)
    motif_bank = MotifBank(max_items=128)
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    rows = []
    schedule = ("explore", "operate", "niche_sort", "stress_gate", "replay")
    phase_index = 0
    phase_age = 0
    phase_len = 4
    stress_pass_rate = 0.0
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = schedule[phase_index]

        if phase == "explore":
            scores = []
            for idx, agent in enumerate(population):
                if len(motif_bank) and rng.random() < 0.70:
                    motif_bank.reinforce(agent, rng, passes=2, reward=0.06)
                rollout_scores = []
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.42,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.04,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    motif_bank.add_success(rollout)
                    novelty = archive.trajectory_novelty(rollout.positions)
                    coverage = len(set(rollout.positions)) / args.max_steps
                    speed = 1.0 - min(rollout.steps, args.max_steps) / args.max_steps
                    rollout_score = 0.45 * novelty + 0.20 * coverage + 0.20 * float(rollout.success) + 0.15 * speed
                    niche_archive.add(agent, rollout, rollout_score, args.max_steps)
                    rollout_scores.append(rollout_score)
                scores.append(float(np.mean(rollout_scores)))
            population = evolve_population(population, scores, rng, mutation_scale=0.10, mutation_rate=0.11)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        elif phase == "operate":
            epsilon = max(0.05, 0.22 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 3):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    motif_bank.add_success(rollout)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            for agent, result in zip(population, results, strict=True):
                niche_archive.add(agent, result.best_rollout, result.score, args.max_steps)
            scores = [exploitation_score(result, args.max_steps) for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        elif phase == "niche_sort":
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=4) for agent in population]
            scores = []
            for agent, result in zip(population, results, strict=True):
                crowding = niche_archive.crowding_penalty(result.best_rollout, args.max_steps)
                niche_quality = result.score + 0.05 * result.success_rate - 0.03 * crowding
                niche_archive.add(agent, result.best_rollout, niche_quality, args.max_steps)
                scores.append(niche_quality)
            local_survivors = niche_archive.local_survivors(per_cell=1, limit=max(2, args.population // 3))
            population = evolve_population(population, scores, rng, elite_frac=0.30, mutation_scale=0.045, mutation_rate=0.06)
            population[: len(local_survivors)] = local_survivors[: len(population)]

        elif phase == "stress_gate":
            normal_results = [_evaluate_minigrid(args, agent, rng, eval_episodes=4) for agent in population]
            stress_results = [_evaluate_minigrid_stress(args, agent, rng, gen, eval_episodes=4) for agent in population]
            scores = []
            passed = []
            for agent, result, stress_result in zip(population, normal_results, stress_results, strict=True):
                niche_quality = 1.0 - niche_archive.crowding_penalty(result.best_rollout, args.max_steps)
                scores.append(speciated_selection_score(result, stress_result, niche_quality, args.max_steps))
                if stress_result.success_rate > 0.0:
                    passed.append(agent.clone())
                    replay_bank.add(stress_result.best_rollout)
                    motif_bank.add_success(stress_result.best_rollout)
            stress_pass_rate = len(passed) / max(1, len(population))
            if not passed:
                passed = [agent.clone() for agent in hall_of_fame[:1]]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.30, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(passed)] = passed[: len(population)]

        else:
            for agent in population:
                replay_bank.reinforce(agent, rng, passes=6, reward=0.10)
                motif_bank.reinforce(agent, rng, passes=3, reward=0.07)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) + 0.04 * result.success_rate for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.35, mutation_scale=0.02, mutation_rate=0.03)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_index, phase_age = _advance_phase(schedule, phase_index, phase_age, phase_len)

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(
                _row(
                    method,
                    gen,
                    best,
                    archive,
                    schedule[phase_index],
                    active_niches=len(niche_archive),
                    stress_pass_rate=stress_pass_rate,
                    replay_bank_size=len(replay_bank),
                    motif_count=len(motif_bank),
                )
            )
    env.close()
    return rows


def _run_cyclic_motif_oriented_radiation(args, rng):
    method = "cyclic_motif_oriented_radiation"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank(max_items=180)
    niche_archive = NicheArchive(max_per_cell=2)
    motif_bank = MotifBank(max_items=160)
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    rows = []
    schedule = ("anchored_explore", "operate", "free_explore", "operate", "replay")
    phase_index = 0
    phase_age = 0
    phase_len = 4
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = schedule[phase_index]

        if phase in {"anchored_explore", "free_explore"}:
            anchored = phase == "anchored_explore"
            scores = []
            for idx, agent in enumerate(population):
                if anchored and len(motif_bank):
                    motif_bank.reinforce(agent, rng, passes=4, reward=0.075)
                rollout_scores = []
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.32 if anchored else 0.48,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.025 if anchored else 0.045,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    motif_bank.add_success(rollout)
                    niche_archive.add(agent, rollout, _trajectory_quality(rollout, args.max_steps), args.max_steps)
                    novelty = archive.trajectory_novelty(rollout.positions)
                    rollout_scores.append(motif_selection_score(_rollout_eval_proxy(rollout, args.max_steps), len(motif_bank), novelty, args.max_steps))
                scores.append(float(np.mean(rollout_scores)))
            population = evolve_population(
                population,
                scores,
                rng,
                mutation_scale=0.065 if anchored else 0.12,
                mutation_rate=0.075 if anchored else 0.12,
            )
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        elif phase == "operate":
            epsilon = max(0.05, 0.22 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 4):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    motif_bank.add_success(rollout)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = []
            for agent, result in zip(population, results, strict=True):
                novelty = archive.trajectory_novelty(result.best_rollout.positions)
                niche_archive.add(agent, result.best_rollout, result.score, args.max_steps)
                scores.append(motif_selection_score(result, len(motif_bank), novelty, args.max_steps))
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        else:
            for agent in population:
                replay_bank.reinforce(agent, rng, passes=5, reward=0.10)
                motif_bank.reinforce(agent, rng, passes=5, reward=0.08)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=8) for agent in population]
            scores = [exploitation_score(result, args.max_steps) + 0.04 * min(1.0, len(motif_bank) / 64.0) for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.35, mutation_scale=0.02, mutation_rate=0.03)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_index, phase_age = _advance_phase(schedule, phase_index, phase_age, phase_len)

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(
                _row(
                    method,
                    gen,
                    best,
                    archive,
                    schedule[phase_index],
                    active_niches=len(niche_archive),
                    replay_bank_size=len(replay_bank),
                    motif_count=len(motif_bank),
                )
            )
    env.close()
    return rows


def _run_cyclic_motif_fast_replay(args, rng):
    method = "cyclic_motif_fast_replay"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank(max_items=160)
    niche_archive = NicheArchive(max_per_cell=2)
    motif_bank = MotifBank(max_items=160)
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    rows = []
    schedule = ("anchored_explore", "operate", "free_explore", "operate", "fast_replay")
    phase_index = 0
    phase_age = 0
    phase_len = 4
    fast_success_count = 0
    total_success_count = 0
    first_success_generation = -1
    best_score_generation = -1
    best_seen_score = -1.0
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = schedule[phase_index]

        if phase in {"anchored_explore", "free_explore"}:
            anchored = phase == "anchored_explore"
            scores = []
            for idx, agent in enumerate(population):
                if anchored and len(motif_bank):
                    motif_bank.reinforce(agent, rng, passes=4, reward=0.08)
                rollout_scores = []
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.28 if anchored else 0.44,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.020 if anchored else 0.040,
                    )
                    archive.add(rollout.positions, rollout.success)
                    added_fast = _add_fast_success(args, replay_bank, motif_bank, rollout)
                    fast_success_count += int(added_fast)
                    total_success_count += int(rollout.success)
                    niche_archive.add(agent, rollout, _fast_rollout_score(rollout, archive, len(motif_bank), args.max_steps), args.max_steps)
                    novelty = archive.trajectory_novelty(rollout.positions)
                    proxy = _rollout_eval_proxy(rollout, args.max_steps)
                    speed = 1.0 - min(proxy.avg_steps, args.max_steps) / args.max_steps
                    rollout_scores.append(
                        0.35 * proxy.success_rate
                        + 0.35 * speed
                        + 0.15 * min(1.0, len(motif_bank) / 32.0)
                        + 0.10 * novelty
                        + 0.05 * proxy.stability
                    )
                scores.append(float(np.mean(rollout_scores)))
            population = evolve_population(
                population,
                scores,
                rng,
                mutation_scale=0.055 if anchored else 0.10,
                mutation_rate=0.065 if anchored else 0.10,
            )
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        elif phase == "operate":
            epsilon = max(0.04, 0.18 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 5):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    archive.add(rollout.positions, rollout.success)
                    added_fast = _add_fast_success(args, replay_bank, motif_bank, rollout)
                    fast_success_count += int(added_fast)
                    total_success_count += int(rollout.success)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = []
            for agent, result in zip(population, results, strict=True):
                novelty = archive.trajectory_novelty(result.best_rollout.positions)
                niche_archive.add(agent, result.best_rollout, result.score, args.max_steps)
                scores.append(_fast_result_score(result, len(motif_bank), novelty, args.max_steps))
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.02, mutation_rate=0.035)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        else:
            for agent in population:
                replay_bank.reinforce(agent, rng, passes=7, reward=0.12)
                motif_bank.reinforce(agent, rng, passes=5, reward=0.09)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=8) for agent in population]
            scores = [_fast_result_score(result, len(motif_bank), 0.0, args.max_steps) + 0.04 * result.success_rate for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.35, mutation_scale=0.018, mutation_rate=0.03)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_index, phase_age = _advance_phase(schedule, phase_index, phase_age, phase_len)

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            if best.success_rate > 0.0 and first_success_generation < 0:
                first_success_generation = gen
            if best.score > best_seen_score:
                best_seen_score = best.score
                best_score_generation = gen
            rows.append(
                _row(
                    method,
                    gen,
                    best,
                    archive,
                    schedule[phase_index],
                    active_niches=len(niche_archive),
                    replay_bank_size=len(replay_bank),
                    motif_count=len(motif_bank),
                    fast_success_count=fast_success_count,
                    fast_replay_ratio=fast_success_count / max(1, total_success_count),
                    first_success_generation=first_success_generation,
                    best_score_generation=best_score_generation,
                )
            )
    env.close()
    return rows


def _run_cyclic_resource_ecology_replay(args, rng):
    method = "cyclic_resource_ecology_replay"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    n_states = probe.n_states
    n_actions = probe.n_actions
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank(max_items=180)
    niche_archive = NicheArchive(max_per_cell=3)
    motif_bank = MotifBank(max_items=128)
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    rows = []
    schedule = ("explore", "operate", "resource_compete", "stress_gate", "reradiate", "replay")
    phase_index = 0
    phase_age = 0
    phase_len = 4
    stress_pass_rate = 0.0
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = schedule[phase_index]

        if phase == "explore":
            scores = []
            for idx, agent in enumerate(population):
                rollout_scores = []
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.45,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.045,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    motif_bank.add_success(rollout)
                    novelty = archive.trajectory_novelty(rollout.positions)
                    crowding = niche_archive.crowding_penalty(rollout, args.max_steps)
                    score = resource_selection_score(
                        _trajectory_quality(rollout, args.max_steps),
                        float(rollout.success),
                        novelty,
                        1.0 - crowding,
                        crowding,
                    )
                    niche_archive.add(agent, rollout, score, args.max_steps)
                    rollout_scores.append(score)
                scores.append(float(np.mean(rollout_scores)))
            population = evolve_population(population, scores, rng, mutation_scale=0.12, mutation_rate=0.12)

        elif phase == "operate":
            epsilon = max(0.05, 0.22 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 3):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
                    motif_bank.add_success(rollout)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) for result in results]
            for agent, result, score in zip(population, results, scores, strict=True):
                niche_archive.add(agent, result.best_rollout, score, args.max_steps)
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        elif phase == "resource_compete":
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=4) for agent in population]
            stress_results = [_evaluate_minigrid_stress(args, agent, rng, gen, eval_episodes=3) for agent in population]
            scores = []
            for agent, result, stress_result in zip(population, results, stress_results, strict=True):
                novelty = archive.trajectory_novelty(result.best_rollout.positions)
                crowding = niche_archive.crowding_penalty(result.best_rollout, args.max_steps)
                score = resource_selection_score(
                    result.score,
                    stress_score(stress_result, args.max_steps),
                    novelty,
                    1.0 - crowding,
                    crowding,
                )
                niche_archive.add(agent, result.best_rollout, score, args.max_steps)
                scores.append(score)
            local_survivors = niche_archive.local_survivors(per_cell=1, limit=max(2, args.population // 2))
            population = evolve_population(population, scores, rng, elite_frac=0.30, mutation_scale=0.055, mutation_rate=0.07)
            population[: len(local_survivors)] = local_survivors[: len(population)]

        elif phase == "stress_gate":
            stress_results = [_evaluate_minigrid_stress(args, agent, rng, gen, eval_episodes=4) for agent in population]
            scores = [stress_score(result, args.max_steps) for result in stress_results]
            passed = []
            for agent, result in zip(population, stress_results, strict=True):
                if result.success_rate > 0.0:
                    passed.append(agent.clone())
                    replay_bank.add(result.best_rollout)
                    motif_bank.add_success(result.best_rollout)
            stress_pass_rate = len(passed) / max(1, len(population))
            if not passed:
                passed = niche_archive.best(max(1, args.population // 8)) or [agent.clone() for agent in hall_of_fame[:1]]
            population = evolve_population(population, scores, rng, elite_frac=0.30, mutation_scale=0.035, mutation_rate=0.05)
            population[: len(passed)] = passed[: len(population)]

        elif phase == "reradiate":
            seeded = niche_archive.sample(rng, max(2, args.population // 2))
            population[: len(seeded)] = seeded[: len(population)]
            while len(population) < args.population:
                population.append(QAgent(n_states, n_actions, rng))
            for agent in population:
                if len(motif_bank) and rng.random() < 0.50:
                    motif_bank.reinforce(agent, rng, passes=2, reward=0.06)
                agent.mutate(scale=0.09, rate=0.10)

        else:
            for agent in population:
                replay_bank.reinforce(agent, rng, passes=5, reward=0.10)
                motif_bank.reinforce(agent, rng, passes=3, reward=0.07)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) + 0.03 * result.success_rate for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, elite_frac=0.35, mutation_scale=0.02, mutation_rate=0.03)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_index, phase_age = _advance_phase(schedule, phase_index, phase_age, phase_len)

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(
                _row(
                    method,
                    gen,
                    best,
                    archive,
                    schedule[phase_index],
                    active_niches=len(niche_archive),
                    stress_pass_rate=stress_pass_rate,
                    replay_bank_size=len(replay_bank),
                    motif_count=len(motif_bank),
                )
            )
    env.close()
    return rows


def _run_cyclic_three_phase(args, rng):
    method = "cyclic_three_phase"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    hall_of_fame = [agent.clone() for agent in population[: max(2, args.population // 6)]]
    archive = NoveltyArchive(size=args.max_steps)
    replay_bank = SuccessReplayBank()
    rows = []
    phases = ("novelty", "replay", "exploit")
    phase_index = 0
    phase_age = 0
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        phase = phases[phase_index]
        if phase == "novelty":
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.35,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                        novelty_archive=archive,
                        novelty_weight=0.03,
                    )
                    archive.add(rollout.positions, rollout.success)
                    replay_bank.add(rollout)
            scores = [_hidden_score(args, agent, archive, rng) for agent in population]
            population = evolve_population(population, scores, rng, mutation_scale=0.10, mutation_rate=0.10)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
        elif phase == "replay":
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent):
                    rollout = _run_minigrid_episode(
                        args,
                        env,
                        agent,
                        rng,
                        epsilon=0.18,
                        train=True,
                        generation=gen,
                        episode=idx * 100 + episode,
                    )
                    replay_bank.add(rollout)
                replay_bank.reinforce(agent, rng, passes=4, reward=0.08)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [0.80 * result.success_rate + 0.20 * result.score for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.04, mutation_rate=0.05)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]
        else:
            epsilon = max(0.05, 0.22 * (1.0 - gen / args.generations))
            for idx, agent in enumerate(population):
                for episode in range(args.episodes_per_agent + 4):
                    rollout = _run_minigrid_episode(args, env, agent, rng, epsilon, True, gen, idx * 100 + episode)
                    replay_bank.add(rollout)
                replay_bank.reinforce(agent, rng, passes=2)
            results = [_evaluate_minigrid(args, agent, rng, eval_episodes=6) for agent in population]
            scores = [exploitation_score(result, args.max_steps) for result in results]
            hall_of_fame = _update_hof_minigrid(args, hall_of_fame, population, rng)
            population = evolve_population(population, scores, rng, mutation_scale=0.025, mutation_rate=0.04)
            population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

        phase_age += 1
        if phase_age >= 6:
            phase_index = (phase_index + 1) % len(phases)
            phase_age = 0

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, agent, rng) for agent in population + hall_of_fame), key=lambda r: r.score)
            rows.append(_row(method, gen, best, archive, phases[phase_index]))
    env.close()
    return rows


@dataclass(frozen=True)
class GoExploreCell:
    states: tuple[int, ...]
    actions: tuple[int, ...]
    score: float
    success: bool
    steps: int


def _run_go_explore_lite(args, rng):
    method = "go_explore_lite"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    robust_agent = QAgent(probe.n_states, probe.n_actions, rng)
    probe.close()
    env = MiniGridTabularEnv(spec)
    archive = NoveltyArchive(size=args.max_steps)
    cells: dict[tuple[int, int], GoExploreCell] = {}
    rows = []

    start = _run_action_sequence(args, env, [], generation=0, episode=0)
    cells[start.positions[-1]] = GoExploreCell(tuple(start.states), (), start.total_reward, start.success, start.steps)
    archive.add(start.positions, start.success)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        attempts = args.population * args.episodes_per_agent
        for episode in range(attempts):
            base = _sample_go_explore_cell(cells, rng)
            extra_len = int(rng.integers(4, min(32, args.max_steps) + 1))
            actions = list(base.actions)
            actions.extend(int(rng.integers(env.n_actions)) for _ in range(extra_len))
            rollout = _run_action_sequence(args, env, actions, generation=gen, episode=episode)
            archive.add(rollout.positions, rollout.success)
            cell_key = rollout.positions[-1]
            quality = _trajectory_quality(rollout, args.max_steps) + 0.002 * len(set(rollout.positions))
            existing = cells.get(cell_key)
            if existing is None or quality > existing.score:
                cells[cell_key] = GoExploreCell(
                    tuple(rollout.states),
                    tuple(rollout.actions),
                    quality,
                    rollout.success,
                    rollout.steps,
                )
                if rollout.success:
                    _reinforce_action_trace(robust_agent, rollout.states, rollout.actions, reward=0.12)
            if rollout.success:
                _reinforce_action_trace(robust_agent, rollout.states, rollout.actions, reward=0.08)

        if gen % args.eval_every == 0 or gen == args.generations:
            result = _evaluate_go_explore(args, robust_agent, list(cells.values()))
            rows.append(_row(method, gen, result, archive, "return_then_explore"))
    env.close()
    return rows


@dataclass
class MapElite:
    agent: QAgent
    score: float


def _run_map_elites_lite(args, rng):
    method = "map_elites_lite"
    spec = _spec(args)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    n_states = probe.n_states
    n_actions = probe.n_actions
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    elites: dict[tuple[int, int, int], MapElite] = {}
    rows = []
    env = MiniGridTabularEnv(spec)

    for gen in range(args.generations + 1):
        if _time_expired(args):
            break
        candidates = []
        if not elites:
            candidates = [QAgent(n_states, n_actions, rng) for _ in range(args.population)]
        else:
            stored = list(elites.values())
            for _ in range(args.population):
                parent = stored[int(rng.integers(len(stored)))].agent
                child = parent.clone()
                child.mutate(scale=0.08, rate=0.08)
                candidates.append(child)

        for idx, agent in enumerate(candidates):
            for episode in range(args.episodes_per_agent):
                rollout = _run_minigrid_episode(args, env, agent, rng, 0.28, True, gen, idx * 100 + episode)
                archive.add(rollout.positions, rollout.success)
            result = _evaluate_minigrid(args, agent, rng, eval_episodes=4)
            descriptor = _map_elites_descriptor(result.best_rollout, args.max_steps)
            existing = elites.get(descriptor)
            if existing is None or result.score > existing.score:
                elites[descriptor] = MapElite(agent.clone(), result.score)

        if gen % args.eval_every == 0 or gen == args.generations:
            best = max((_evaluate_minigrid(args, elite.agent, rng) for elite in elites.values()), key=lambda r: r.score)
            rows.append(_row(method, gen, best, archive, f"cells={len(elites)}"))
    env.close()
    return rows


def _run_minigrid_episode(
    args,
    env,
    agent,
    rng,
    epsilon,
    train,
    generation,
    episode,
    novelty_archive=None,
    novelty_weight=0.0,
) -> Rollout:
    env.episode_seed = args.seed + generation * 10000 + episode
    state = env.reset()
    states = [state]
    positions = [env.position]
    actions = []
    rewards = []
    success = False
    while True:
        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)
        if novelty_archive is not None:
            reward += novelty_weight * novelty_archive.position_bonus(info["position"])
        if train:
            agent.update(state, action, reward, next_state, done)
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(info["position"])
        state = next_state
        success = bool(info["success"])
        if done:
            break
    return Rollout(states, positions, actions, rewards, success, len(actions), float(np.sum(rewards)))


def _run_action_sequence(args, env, actions, generation, episode) -> Rollout:
    env.episode_seed = args.seed + generation * 10000 + episode
    state = env.reset()
    states = [state]
    positions = [env.position]
    taken_actions = []
    rewards = []
    success = False
    done = False
    for action in actions[: args.max_steps]:
        next_state, reward, done, info = env.step(action)
        taken_actions.append(int(action))
        rewards.append(reward)
        states.append(next_state)
        positions.append(info["position"])
        state = next_state
        success = bool(info["success"])
        if done:
            break
    while not done and len(taken_actions) < args.max_steps:
        action = 2
        next_state, reward, done, info = env.step(action)
        taken_actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(info["position"])
        state = next_state
        success = bool(info["success"])
        if done:
            break
    return Rollout(states, positions, taken_actions, rewards, success, len(taken_actions), float(np.sum(rewards)))


def _evaluate_minigrid(args, agent, rng, eval_episodes=None):
    eval_episodes = eval_episodes or args.eval_episodes
    spec = _spec(args)
    env = MiniGridTabularEnv(spec)
    rollouts = [
        _run_minigrid_episode(args, env, agent, rng, 0.0, False, 9000, idx)
        for idx in range(eval_episodes)
    ]
    env.close()
    successes = [rollout for rollout in rollouts if rollout.success]
    success_rate = len(successes) / len(rollouts)
    avg_steps = float(np.mean([rollout.steps for rollout in successes])) if successes else float(args.max_steps)
    speed = 1.0 - min(avg_steps, args.max_steps) / args.max_steps
    stability = 0.0
    score = 0.55 * success_rate + 0.30 * speed + 0.15 * stability
    best_rollout = min(rollouts, key=lambda rollout: (not rollout.success, rollout.steps))
    return EvalResult(success_rate, avg_steps, stability, score, best_rollout)


def _rollout_eval_proxy(rollout: Rollout, max_steps: int) -> EvalResult:
    success_rate = 1.0 if rollout.success else 0.0
    avg_steps = float(rollout.steps if rollout.success else max_steps)
    speed = 1.0 - min(avg_steps, max_steps) / max_steps
    score = 0.55 * success_rate + 0.30 * speed
    return EvalResult(success_rate, avg_steps, 0.0, score, rollout)


def _fast_success_threshold(max_steps: int) -> int:
    return max(16, int(max_steps * 0.12))


def _add_fast_success(args, replay_bank: SuccessReplayBank, motif_bank: MotifBank, rollout: Rollout) -> bool:
    if not rollout.success:
        return False
    if rollout.steps > _fast_success_threshold(args.max_steps):
        return False
    replay_bank.add(rollout)
    motif_bank.add_success(rollout)
    return True


def _fast_result_score(result: EvalResult, motif_count: int, novelty: float, max_steps: int) -> float:
    speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
    motif_reuse = min(1.0, motif_count / 32.0)
    success_gate = result.success_rate**1.5
    return 0.36 * success_gate + 0.38 * speed + 0.12 * motif_reuse + 0.09 * novelty + 0.05 * result.stability


def _fast_rollout_score(rollout: Rollout, archive: NoveltyArchive, motif_count: int, max_steps: int) -> float:
    novelty = archive.trajectory_novelty(rollout.positions)
    proxy = _rollout_eval_proxy(rollout, max_steps)
    return _fast_result_score(proxy, motif_count, novelty, max_steps)


def _evaluate_minigrid_stress(args, agent, rng, generation: int, eval_episodes: int = 4):
    stress_steps = max(24, int(args.max_steps * 0.65))
    spec = MiniGridSpec(args.env_id, stress_steps, args.seed + 5000 + generation, args.state_encoder)
    env = MiniGridTabularEnv(spec)
    rollouts = [
        _run_minigrid_episode(args, env, agent, rng, 0.03, False, 9100 + generation, idx)
        for idx in range(eval_episodes)
    ]
    env.close()
    successes = [rollout for rollout in rollouts if rollout.success]
    success_rate = len(successes) / len(rollouts)
    avg_steps = float(np.mean([rollout.steps for rollout in successes])) if successes else float(stress_steps)
    speed = 1.0 - min(avg_steps, stress_steps) / stress_steps
    score = 0.55 * success_rate + 0.30 * speed
    best_rollout = min(rollouts, key=lambda rollout: (not rollout.success, rollout.steps))
    return EvalResult(success_rate, avg_steps, 0.0, score, best_rollout)


def _evaluate_go_explore(args, robust_agent, cells):
    if not cells:
        raise ValueError("go-explore archive is empty")
    policy_result = _evaluate_minigrid(args, robust_agent, np.random.default_rng(args.seed + 991))
    if policy_result.success_rate > 0.0:
        return policy_result
    best_cell = max(cells, key=lambda cell: (cell.success, cell.score, -cell.steps))
    spec = _spec(args)
    env = MiniGridTabularEnv(spec)
    rollouts = [_run_action_sequence(args, env, best_cell.actions, 9000, idx) for idx in range(args.eval_episodes)]
    env.close()
    successes = [rollout for rollout in rollouts if rollout.success]
    success_rate = len(successes) / len(rollouts)
    avg_steps = float(np.mean([rollout.steps for rollout in successes])) if successes else float(args.max_steps)
    speed = 1.0 - min(avg_steps, args.max_steps) / args.max_steps
    score = 0.55 * success_rate + 0.30 * speed
    best_rollout = min(rollouts, key=lambda rollout: (not rollout.success, rollout.steps))
    return EvalResult(success_rate, avg_steps, 0.0, score, best_rollout)


def _hidden_score(args, agent, archive, rng):
    result = _evaluate_minigrid(args, agent, rng, eval_episodes=4)
    novelty = archive.trajectory_novelty(result.best_rollout.positions)
    speed = 1.0 - min(result.avg_steps, args.max_steps) / args.max_steps
    return 0.30 * result.success_rate + 0.25 * speed + 0.25 * novelty + 0.20 * result.stability


def _update_hof_minigrid(args, hall_of_fame, population, rng):
    candidates = hall_of_fame + population
    ranked = sorted(
        candidates,
        key=lambda agent: _evaluate_minigrid(args, agent, rng, eval_episodes=6).score,
        reverse=True,
    )
    return [agent.clone() for agent in ranked[: max(2, args.population // 4)]]


def _sample_go_explore_cell(cells, rng):
    values = list(cells.values())
    weights = np.array([max(0.05, cell.score + (0.25 if cell.success else 0.0)) for cell in values], dtype=np.float64)
    weights = weights / weights.sum()
    return values[int(rng.choice(len(values), p=weights))]


def _trajectory_quality(rollout: Rollout, max_steps: int) -> float:
    speed = 1.0 - min(rollout.steps, max_steps) / max_steps
    return (1.0 if rollout.success else 0.0) + 0.20 * speed + rollout.total_reward


def _stress_score(result: EvalResult, max_steps: int) -> float:
    return stress_score(result, max_steps)


def _ecology_bottleneck(args, population, stress_survivors, niche_archive, hall_of_fame, rng, n_states):
    next_population = []
    survivor_count = max(1, int(args.population * 0.40))
    niche_count = max(1, int(args.population * 0.30))
    hof_count = max(1, int(args.population * 0.20))

    for agent in stress_survivors[:survivor_count]:
        next_population.append(agent.clone())
    next_population.extend(niche_archive.sample(rng, niche_count))
    for agent in hall_of_fame[:hof_count]:
        child = agent.clone()
        child.mutate(scale=0.06, rate=0.07)
        next_population.append(child)

    while len(next_population) < args.population:
        if population and rng.random() < 0.5:
            child = population[int(rng.integers(len(population)))].clone()
            child.mutate(scale=0.12, rate=0.12)
            next_population.append(child)
        else:
            next_population.append(QAgent(n_states, 3, rng))
    return next_population[: args.population]


def _reinforce_action_trace(agent: QAgent, states: list[int] | tuple[int, ...], actions: list[int] | tuple[int, ...], reward: float) -> None:
    reinforce_action_trace(agent, states, actions, reward)


def _map_elites_descriptor(rollout: Rollout, max_steps: int) -> tuple[int, int, int]:
    row, col = rollout.positions[-1]
    row_bin = min(7, max(0, row // 2))
    col_bin = min(7, max(0, col // 2))
    speed_bin = min(3, int(4 * (1.0 - min(rollout.steps, max_steps) / max_steps)))
    return row_bin, col_bin, speed_bin


def _behavior_descriptor(rollout: Rollout, max_steps: int) -> tuple[int, ...]:
    return behavior_descriptor(rollout, max_steps)


def _most_mobile_window(positions: list[tuple[int, int]], window: int) -> int:
    return most_mobile_window(positions, window)


def _row(
    method,
    generation,
    result,
    archive,
    phase,
    active_niches: int = 0,
    stress_pass_rate: float = 0.0,
    replay_bank_size: int = 0,
    motif_count: int = 0,
    fast_success_count: int = 0,
    fast_replay_ratio: float = 0.0,
    first_success_generation: int = -1,
    best_score_generation: int = -1,
):
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
        "active_niches": active_niches,
        "stress_pass_rate": round(stress_pass_rate, 4),
        "replay_bank_size": replay_bank_size,
        "motif_count": motif_count,
        "fast_success_count": fast_success_count,
        "fast_replay_ratio": round(fast_replay_ratio, 4),
        "first_success_generation": first_success_generation,
        "best_score_generation": best_score_generation,
    }


def _spec(args) -> MiniGridSpec:
    return MiniGridSpec(args.env_id, args.max_steps, args.seed, args.state_encoder)


if __name__ == "__main__":
    main()
