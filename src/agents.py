from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env import GridWorld


@dataclass
class Rollout:
    states: list[int]
    positions: list[tuple[int, int]]
    actions: list[int]
    rewards: list[float]
    success: bool
    steps: int
    total_reward: float


class QAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        rng: np.random.Generator,
        alpha: float = 0.25,
        gamma: float = 0.96,
    ):
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng

    def clone(self) -> "QAgent":
        child_seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
        child = QAgent(self.q.shape[0], self.q.shape[1], np.random.default_rng(child_seed), self.alpha, self.gamma)
        child.q = self.q.copy()
        return child

    def clone_with_seed(self, seed: int) -> "QAgent":
        child = QAgent(self.q.shape[0], self.q.shape[1], np.random.default_rng(seed), self.alpha, self.gamma)
        child.q = self.q.copy()
        return child

    def act(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        values = self.q[state]
        max_value = values.max()
        choices = np.flatnonzero(values == max_value)
        return int(self.rng.choice(choices))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward
        if not done:
            target += self.gamma * float(self.q[next_state].max())
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def mutate(self, scale: float = 0.08, rate: float = 0.08) -> None:
        mask = self.rng.random(self.q.shape) < rate
        noise = self.rng.normal(0.0, scale, size=self.q.shape)
        self.q = self.q + mask * noise


def crossover(parent_a: QAgent, parent_b: QAgent, rng: np.random.Generator) -> QAgent:
    child = parent_a.clone()
    mask = rng.random(child.q.shape) < 0.5
    child.q[mask] = parent_b.q[mask]
    return child


def run_episode(
    env: GridWorld,
    agent: QAgent,
    epsilon: float,
    train: bool,
    novelty_bonus=None,
) -> Rollout:
    state = env.reset()
    states = [state]
    positions = [env.position]
    actions: list[int] = []
    rewards: list[float] = []
    success = False

    while True:
        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)
        if novelty_bonus is not None:
            reward += float(novelty_bonus(next_state, info["position"]))
        if train:
            agent.update(state, action, reward, next_state, done)
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(env.position)
        state = next_state
        success = bool(info["success"])
        if done:
            break

    return Rollout(
        states=states,
        positions=positions,
        actions=actions,
        rewards=rewards,
        success=success,
        steps=env.steps,
        total_reward=float(np.sum(rewards)),
    )
