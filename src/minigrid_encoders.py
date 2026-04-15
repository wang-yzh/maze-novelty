from __future__ import annotations

from typing import Protocol

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX


class MiniGridStateEncoder(Protocol):
    name: str

    @property
    def n_states(self) -> int:
        ...

    def encode(self, obs: dict, carrying: bool, last_action: int) -> int:
        ...


class MiniGridCompactEncoder:
    name = "compact"

    @property
    def n_states(self) -> int:
        return 4 * 11 * 11 * 11 * 2 * 2 * 2 * 2 * 2

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

    def encode(self, obs: dict, carrying: bool, last_action: int) -> int:
        image = obs["image"]
        local_shape = (
            _cell_category(image, 3, 5),  # front
            _cell_category(image, 2, 5),  # front-left
            _cell_category(image, 4, 5),  # front-right
            _cell_category(image, 2, 6),  # left
            _cell_category(image, 4, 6),  # right
        )
        features = (
            int(obs["direction"]),
            _mixed_radix(local_shape, (4, 4, 4, 4, 4)),
            _goal_relative_bin(image),
            _topology_type(local_shape),
            min(last_action, 3),
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
