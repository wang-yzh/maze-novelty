from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from minigrid_encoders import MiniGridStateEncoder, StateSignature, make_minigrid_encoder


Position = tuple[int, int]


@dataclass(frozen=True)
class MiniGridSpec:
    env_id: str
    max_steps: int = 256
    seed: int = 7
    state_encoder: str = "compact"


class MiniGridTabularEnv:
    """Small adapter from MiniGrid observations to compact tabular features."""

    action_map = (0, 1, 2)  # left, right, forward

    def __init__(self, spec: MiniGridSpec, episode_seed: int | None = None, encoder: MiniGridStateEncoder | None = None):
        self.spec = spec
        self.env = gym.make(spec.env_id, max_steps=spec.max_steps)
        self.episode_seed = episode_seed
        self.encoder = encoder or make_minigrid_encoder(spec.state_encoder)
        self.last_obs = None
        self.last_signature = None
        self.last_action = self.n_actions
        self.steps = 0

    @property
    def n_states(self) -> int:
        return self.encoder.n_states

    @property
    def n_actions(self) -> int:
        return len(self.action_map)

    @property
    def position(self) -> Position:
        x, y = getattr(self.env.unwrapped, "agent_pos")
        return int(y), int(x)

    @property
    def state_signature(self) -> StateSignature:
        if self.last_obs is None or self.last_signature is None:
            msg = "environment has no current observation signature; call reset() first"
            raise ValueError(msg)
        return self.last_signature

    def reset(self) -> int:
        obs, _info = self.env.reset(seed=self.episode_seed)
        self.last_obs = obs
        self.last_action = self.n_actions
        self.steps = 0
        self.last_signature = self.observation_signature(obs, self.last_action)
        return self.observation_index(obs, self.last_action)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        minigrid_action = self.action_map[action]
        obs, reward, terminated, truncated, info = self.env.step(minigrid_action)
        self.last_obs = obs
        self.last_action = action
        self.last_signature = self.observation_signature(obs, self.last_action)
        self.steps += 1
        done = terminated or truncated
        reward_value = float(reward)
        success = bool(terminated and reward_value > 0.0)
        shaped_reward = reward_value - 0.005
        return self.observation_index(obs, self.last_action), shaped_reward, done, {
            "success": success,
            "steps": self.steps,
            "position": self.position,
            "state_signature": self.last_signature,
        }

    def observation_index(self, obs: dict, last_action: int) -> int:
        carrying = self._carrying_flag()
        return self.encoder.encode(obs, bool(carrying), last_action)

    def observation_signature(self, obs: dict, last_action: int) -> StateSignature:
        carrying = self._carrying_flag()
        return self.encoder.signature(obs, bool(carrying), last_action)

    def _carrying_flag(self) -> int:
        return 1 if getattr(self.env.unwrapped, "carrying", None) is not None else 0

    def close(self) -> None:
        self.env.close()
