from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX


Position = tuple[int, int]


@dataclass(frozen=True)
class MiniGridSpec:
    env_id: str
    max_steps: int = 256
    seed: int = 7


class MiniGridTabularEnv:
    """Small adapter from MiniGrid observations to compact tabular features."""

    action_map = (0, 1, 2)  # left, right, forward

    def __init__(self, spec: MiniGridSpec, episode_seed: int | None = None):
        self.spec = spec
        self.env = gym.make(spec.env_id, max_steps=spec.max_steps)
        self.episode_seed = episode_seed
        self.last_obs = None
        self.steps = 0

    @property
    def n_states(self) -> int:
        return 4 * 11 * 11 * 11 * 2 * 2 * 2 * 2 * 2

    @property
    def n_actions(self) -> int:
        return len(self.action_map)

    @property
    def position(self) -> Position:
        x, y = getattr(self.env.unwrapped, "agent_pos")
        return int(y), int(x)

    def reset(self) -> int:
        obs, _info = self.env.reset(seed=self.episode_seed)
        self.last_obs = obs
        self.steps = 0
        return self.observation_index(obs)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        minigrid_action = self.action_map[action]
        obs, reward, terminated, truncated, info = self.env.step(minigrid_action)
        self.last_obs = obs
        self.steps += 1
        done = terminated or truncated
        reward_value = float(reward)
        success = bool(terminated and reward_value > 0.0)
        shaped_reward = reward_value - 0.005
        return self.observation_index(obs), shaped_reward, done, {
            "success": success,
            "steps": self.steps,
            "position": self.position,
        }

    def observation_index(self, obs: dict) -> int:
        image = obs["image"]
        direction = int(obs["direction"])
        front = _cell_type(image, 3, 5)
        left = _cell_type(image, 2, 4)
        right = _cell_type(image, 4, 4)
        seen_goal = _seen(image, "goal")
        seen_key = _seen(image, "key")
        seen_door = _seen(image, "door")
        seen_lava = _seen(image, "lava")
        carrying = 1 if getattr(self.env.unwrapped, "carrying", None) is not None else 0

        features = (
            direction,
            front,
            left,
            right,
            seen_goal,
            seen_key,
            seen_door,
            seen_lava,
            carrying,
        )
        bases = (4, 11, 11, 11, 2, 2, 2, 2, 2)
        index = 0
        for value, base in zip(features, bases, strict=True):
            index = index * base + int(value)
        return index

    def close(self) -> None:
        self.env.close()


def _cell_type(image: np.ndarray, x: int, y: int) -> int:
    return int(image[x, y, 0])


def _seen(image: np.ndarray, object_name: str) -> int:
    return int(np.any(image[:, :, 0] == OBJECT_TO_IDX[object_name]))
