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

from agents import QAgent, Rollout
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
    parser.add_argument("--methods", default="q_learning,genetic_q,cyclic_novelty,cyclic_replay")
    args = parser.parse_args()

    methods = {
        "q_learning": run_q_learning,
        "q_learning_strong": run_q_learning_strong,
        "genetic_q": run_genetic_q,
        "genetic_q_strong": run_genetic_q_strong,
        "cyclic_novelty": run_cyclic_novelty,
        "cyclic_replay": run_cyclic_replay,
        "cyclic_three_phase": run_cyclic_three_phase,
    }
    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    rows = []
    for method in requested:
        print(f"Running {method}...")
        rng = np.random.default_rng(args.seed + METHOD_SEED_OFFSETS[method])
        rows.extend(methods[method](args, rng))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(args.output_dir / "metrics.csv", rows)
    print(f"Done. Outputs written to {args.output_dir}")


def run_q_learning(args, rng):
    return _run_q_learning(args, rng, "q_learning", episode_multiplier=1)


def run_q_learning_strong(args, rng):
    return _run_q_learning(args, rng, "q_learning_strong", episode_multiplier=4)


def _run_q_learning(args, rng, method: str, episode_multiplier: int):
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    agent = QAgent(probe.n_states, probe.n_actions, rng)
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    rows = []
    env = MiniGridTabularEnv(spec)
    for gen in range(args.generations + 1):
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
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
    probe = MiniGridTabularEnv(spec, episode_seed=args.seed)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(args.population)]
    probe.close()
    archive = NoveltyArchive(size=args.max_steps)
    rows = []
    env = MiniGridTabularEnv(spec)
    for gen in range(args.generations + 1):
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


def _run_cyclic(args, rng, method: str, use_replay: bool):
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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


def _run_cyclic_three_phase(args, rng):
    method = "cyclic_three_phase"
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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


def _evaluate_minigrid(args, agent, rng, eval_episodes=None):
    eval_episodes = eval_episodes or args.eval_episodes
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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
