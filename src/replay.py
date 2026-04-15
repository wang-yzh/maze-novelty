from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from agents import QAgent, Rollout


@dataclass(frozen=True)
class ReplayItem:
    states: tuple[int, ...]
    actions: tuple[int, ...]
    steps: int


class SuccessReplayBank:
    def __init__(self, max_items: int = 80):
        self.items: deque[ReplayItem] = deque(maxlen=max_items)

    def add(self, rollout: Rollout) -> None:
        if not rollout.success or not rollout.actions:
            return
        self.items.append(
            ReplayItem(
                states=tuple(rollout.states),
                actions=tuple(rollout.actions),
                steps=rollout.steps,
            )
        )

    def __len__(self) -> int:
        return len(self.items)

    def reinforce(
        self,
        agent: QAgent,
        rng: np.random.Generator,
        passes: int = 2,
        reward: float = 0.08,
    ) -> None:
        if not self.items:
            return
        items = list(self.items)
        for _ in range(passes):
            item = items[int(rng.integers(len(items)))]
            for idx, action in enumerate(item.actions):
                state = item.states[idx]
                next_state = item.states[idx + 1]
                done = idx == len(item.actions) - 1
                shaped_reward = reward
                if done:
                    shaped_reward += 1.0
                agent.update(state, action, shaped_reward, next_state, done)

    def seed_agent(self, base: QAgent, rng: np.random.Generator) -> QAgent:
        child = base.clone()
        self.reinforce(child, rng, passes=4, reward=0.10)
        return child
