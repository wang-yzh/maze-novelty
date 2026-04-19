from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

STRICT_MATCH = "strict"
DIRECTION_AGNOSTIC_MATCH = "direction_agnostic"


@dataclass(frozen=True)
class StateSignature:
    direction: int
    local_shape: tuple[int, int, int, int, int]
    goal_bin: int
    topology: int
    last_action: int

    def matches(self, other: "StateSignature", mode: str = STRICT_MATCH) -> bool:
        goal_compatible = self.goal_bin == other.goal_bin or self.goal_bin == 4 or other.goal_bin == 4
        action_compatible = self.last_action == other.last_action or self.last_action == 3 or other.last_action == 3
        direction_compatible = self._direction_matches(other, mode)
        return (
            direction_compatible
            and self.local_shape == other.local_shape
            and self.topology == other.topology
            and action_compatible
            and goal_compatible
        )

    def canonical_key(self, mode: str = STRICT_MATCH) -> tuple[int, tuple[int, int, int, int, int], int, int, int]:
        direction = self.direction if mode == STRICT_MATCH else -1 if mode == DIRECTION_AGNOSTIC_MATCH else None
        if direction is None:
            msg = f"Unknown StateSignature match mode: {mode}"
            raise ValueError(msg)
        return (direction, self.local_shape, self.goal_bin, self.topology, self.last_action)

    def _direction_matches(self, other: "StateSignature", mode: str) -> bool:
        if mode == STRICT_MATCH:
            return self.direction == other.direction
        if mode == DIRECTION_AGNOSTIC_MATCH:
            return True
        msg = f"Unknown StateSignature match mode: {mode}"
        raise ValueError(msg)


class MiniGridStateEncoder(Protocol):
    name: str

    @property
    def n_states(self) -> int:
        ...

    def signature(self, obs: dict, carrying: bool, last_action: int) -> StateSignature:
        ...

    def encode(self, obs: dict, carrying: bool, last_action: int) -> int:
        ...


class MiniGridCompactEncoder:
    name = "compact"

    @property
    def n_states(self) -> int:
        return 4 * 11 * 11 * 11 * 2 * 2 * 2 * 2 * 2

    def signature(self, obs: dict, carrying: bool, last_action: int) -> StateSignature:
        return _state_signature(obs, last_action)

    def encode(self, obs: dict, carrying: bool, last_action: int) -> int:
        image = obs["image"]
        features = (
            int(obs["direction"]),
            _cell_type(image, 3, 5),
            _cell_type(image, 2, 4),
            _cell_type(image, 4, 4),
            _seen(image, "goal"),
            _seen(image, "key"),
            _seen(image, "door"),
            _seen(image, "lava"),
            int(carrying),
        )
        return _mixed_radix(features, (4, 11, 11, 11, 2, 2, 2, 2, 2))


class MiniGridGeometryEncoder:
    name = "geometry"

    @property
    def n_states(self) -> int:
        return 4 * (4**5) * 9 * 5 * 4

    def signature(self, obs: dict, carrying: bool, last_action: int) -> StateSignature:
        return _state_signature(obs, last_action)

    def encode(self, obs: dict, carrying: bool, last_action: int) -> int:
        signature = self.signature(obs, carrying, last_action)
        features = (
            signature.direction,
            _mixed_radix(signature.local_shape, (4, 4, 4, 4, 4)),
            signature.goal_bin,
            signature.topology,
            signature.last_action,
        )
        return _mixed_radix(features, (4, 4**5, 9, 5, 4))


def make_minigrid_encoder(name: str) -> MiniGridStateEncoder:
    if name == "compact":
        return MiniGridCompactEncoder()
    if name == "geometry":
        return MiniGridGeometryEncoder()
    raise ValueError(f"Unknown MiniGrid state encoder: {name}")


def _mixed_radix(features: tuple[int, ...], bases: tuple[int, ...]) -> int:
    index = 0
    for value, base in zip(features, bases, strict=True):
        index = index * base + int(value)
    return index


def _state_signature(obs: dict, last_action: int) -> StateSignature:
    image = obs["image"]
    local_shape = (
        _cell_category(image, 3, 5),  # front
        _cell_category(image, 2, 5),  # front-left
        _cell_category(image, 4, 5),  # front-right
        _cell_category(image, 2, 6),  # left
        _cell_category(image, 4, 6),  # right
    )
    return StateSignature(
        direction=int(obs["direction"]),
        local_shape=local_shape,
        goal_bin=_goal_relative_bin(image),
        topology=_topology_type(local_shape),
        last_action=min(last_action, 3),
    )


def _cell_type(image: np.ndarray, x: int, y: int) -> int:
    return int(image[x, y, 0])


def _seen(image: np.ndarray, object_name: str) -> int:
    return int(np.any(image[:, :, 0] == OBJECT_TO_IDX[object_name]))


def _cell_category(image: np.ndarray, x: int, y: int) -> int:
    object_type = int(image[x, y, 0])
    if object_type == OBJECT_TO_IDX["wall"]:
        return 1
    if object_type == OBJECT_TO_IDX["goal"]:
        return 2
    if object_type in (OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["floor"], OBJECT_TO_IDX["unseen"]):
        return 0
    return 3


def _goal_relative_bin(image: np.ndarray) -> int:
    goal_positions = np.argwhere(image[:, :, 0] == OBJECT_TO_IDX["goal"])
    if len(goal_positions) == 0:
        return 4
    x, y = goal_positions[0]
    dx = int(np.sign(int(x) - 3)) + 1
    dy = int(np.sign(int(y) - 6)) + 1
    return dx * 3 + dy


def _topology_type(local_shape: tuple[int, ...]) -> int:
    passable = sum(1 for category in local_shape if category in (0, 2))
    front_open = local_shape[0] in (0, 2)
    side_open = local_shape[3] in (0, 2) or local_shape[4] in (0, 2)
    if passable <= 1:
        return 0
    if front_open and not side_open:
        return 1
    if front_open and side_open:
        return 2
    if side_open:
        return 3
    return 4
