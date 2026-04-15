from __future__ import annotations
# ruff: noqa: E402

import argparse
import os
import time
from dataclasses import dataclass
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
    "cyclic_operate_replay": 601,
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
        "go_explore_lite": run_go_explore_lite,
        "map_elites_lite": run_map_elites_lite,
    }
    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    rows = []
    for method in requested:
        print(f"Running {method}...")
        rng = np.random.default_rng(args.seed + METHOD_SEED_OFFSETS[method])
        start_time = time.perf_counter()
        args.method_deadline = start_time + args.method_time_limit_seconds if args.method_time_limit_seconds > 0 else None
        method_rows = methods[method](args, rng)
        runtime_seconds = time.perf_counter() - start_time
        hit_time_limit = args.method_deadline is not None and runtime_seconds >= args.method_time_limit_seconds
        for row in method_rows:
            row["runtime_seconds"] = round(runtime_seconds, 4)
            row["time_limited"] = int(hit_time_limit)
        rows.extend(method_rows)
        print(f"{method} runtime_seconds={runtime_seconds:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(args.output_dir / "metrics.csv", rows)
    print(f"Done. Outputs written to {args.output_dir}")


def _time_expired(args) -> bool:
    deadline = getattr(args, "method_deadline", None)
    return deadline is not None and time.perf_counter() >= deadline


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
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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


def run_go_explore_lite(args, rng):
    return _run_go_explore_lite(args, rng)


def run_map_elites_lite(args, rng):
    return _run_map_elites_lite(args, rng)


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
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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


def _evaluate_go_explore(args, robust_agent, cells):
    if not cells:
        raise ValueError("go-explore archive is empty")
    policy_result = _evaluate_minigrid(args, robust_agent, np.random.default_rng(args.seed + 991))
    if policy_result.success_rate > 0.0:
        return policy_result
    best_cell = max(cells, key=lambda cell: (cell.success, cell.score, -cell.steps))
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed)
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


def _reinforce_action_trace(agent: QAgent, states: list[int] | tuple[int, ...], actions: list[int] | tuple[int, ...], reward: float) -> None:
    for idx, action in enumerate(actions):
        state = states[idx]
        next_state = states[idx + 1]
        done = idx == len(actions) - 1
        shaped_reward = reward + (1.0 if done else 0.0)
        agent.update(state, action, shaped_reward, next_state, done)


def _map_elites_descriptor(rollout: Rollout, max_steps: int) -> tuple[int, int, int]:
    row, col = rollout.positions[-1]
    row_bin = min(7, max(0, row // 2))
    col_bin = min(7, max(0, col // 2))
    speed_bin = min(3, int(4 * (1.0 - min(rollout.steps, max_steps) / max_steps)))
    return row_bin, col_bin, speed_bin


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
