from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents import QAgent, Rollout, crossover, run_episode
from env import GridWorld
from novelty import NoveltyArchive
from replay import SuccessReplayBank


@dataclass
class EvalResult:
    success_rate: float
    avg_steps: float
    stability: float
    score: float
    best_rollout: Rollout


def make_population(
    count: int,
    n_states: int,
    n_actions: int,
    rng: np.random.Generator,
) -> list[QAgent]:
    return [QAgent(n_states, n_actions, rng) for _ in range(count)]


def evaluate_agent(agent: QAgent, grids: list[np.ndarray], max_steps: int) -> EvalResult:
    rollouts: list[Rollout] = []
    path_signatures: list[tuple[tuple[int, int], ...]] = []
    for grid in grids:
        env = GridWorld(grid, max_steps=max_steps)
        rollout = run_episode(env, agent, epsilon=0.0, train=False)
        rollouts.append(rollout)
        path_signatures.append(tuple(rollout.positions))

    successes = [r for r in rollouts if r.success]
    success_rate = len(successes) / len(rollouts)
    avg_steps = float(np.mean([r.steps for r in successes])) if successes else float(max_steps)
    speed = 1.0 - min(avg_steps, max_steps) / max_steps
    stability = _stability(path_signatures)
    score = 0.55 * success_rate + 0.30 * speed + 0.15 * stability
    best_rollout = min(rollouts, key=lambda r: (not r.success, r.steps))
    return EvalResult(success_rate, avg_steps, stability, score, best_rollout)


def exploitation_score(result: EvalResult, max_steps: int) -> float:
    speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
    success_gate = result.success_rate**2
    return 0.62 * success_gate + 0.28 * speed + 0.10 * result.stability


def evolve_population(
    population: list[QAgent],
    scores: list[float],
    rng: np.random.Generator,
    elite_frac: float = 0.25,
    mutation_scale: float = 0.07,
    mutation_rate: float = 0.07,
) -> list[QAgent]:
    elite_count = max(2, int(len(population) * elite_frac))
    order = np.argsort(scores)[::-1]
    elites = [population[int(i)].clone() for i in order[:elite_count]]
    next_population = [elite.clone() for elite in elites]
    while len(next_population) < len(population):
        index_a = int(rng.integers(len(elites)))
        index_b = int(rng.integers(len(elites)))
        parent_a = elites[index_a]
        parent_b = elites[index_b]
        child = crossover(parent_a, parent_b, rng)
        child.mutate(scale=mutation_scale, rate=mutation_rate)
        next_population.append(child)
    return next_population


def train_population_task(
    population: list[QAgent],
    train_grids: list[np.ndarray],
    rng: np.random.Generator,
    episodes_per_agent: int,
    max_steps: int,
    epsilon: float,
) -> None:
    for agent in population:
        for _ in range(episodes_per_agent):
            grid = train_grids[int(rng.integers(len(train_grids)))]
            run_episode(GridWorld(grid, max_steps=max_steps), agent, epsilon=epsilon, train=True)


def train_population_task_with_replay(
    population: list[QAgent],
    train_grids: list[np.ndarray],
    rng: np.random.Generator,
    episodes_per_agent: int,
    max_steps: int,
    epsilon: float,
    replay_bank: SuccessReplayBank | None,
) -> None:
    for agent in population:
        for _ in range(episodes_per_agent):
            grid = train_grids[int(rng.integers(len(train_grids)))]
            rollout = run_episode(GridWorld(grid, max_steps=max_steps), agent, epsilon=epsilon, train=True)
            if replay_bank is not None:
                replay_bank.add(rollout)
        if replay_bank is not None:
            replay_bank.reinforce(agent, rng, passes=2)


def train_population_novelty(
    population: list[QAgent],
    train_grids: list[np.ndarray],
    archive: NoveltyArchive,
    rng: np.random.Generator,
    episodes_per_agent: int,
    max_steps: int,
    epsilon: float,
    bonus_weight: float,
) -> list[float]:
    scores = []
    for agent in population:
        novelty_scores = []
        for _ in range(episodes_per_agent):
            grid = train_grids[int(rng.integers(len(train_grids)))]
            env = GridWorld(grid, max_steps=max_steps)
            rollout = run_episode(
                env,
                agent,
                epsilon=epsilon,
                train=True,
                novelty_bonus=lambda _s, pos: bonus_weight * archive.position_bonus(pos),
            )
            novelty = archive.trajectory_novelty(rollout.positions)
            novelty_scores.append(novelty + (0.15 if rollout.success else 0.0))
            archive.add(rollout.positions, rollout.success)
        scores.append(float(np.mean(novelty_scores)))
    return scores


def _stability(signatures: list[tuple[tuple[int, int], ...]]) -> float:
    if len(signatures) <= 1:
        return 1.0
    unique = len(set(signatures))
    return 1.0 - (unique - 1) / max(1, len(signatures) - 1)
