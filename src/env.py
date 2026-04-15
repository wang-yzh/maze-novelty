from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Action = int
Position = tuple[int, int]


@dataclass(frozen=True)
class MazeSpec:
    size: int = 12
    obstacle_prob: float = 0.22
    max_steps: int = 144


class GridWorld:
    def __init__(self, grid: np.ndarray, max_steps: int | None = None):
        self.grid = grid.astype(np.int8)
        self.size = int(grid.shape[0])
        self.start: Position = (0, 0)
        self.goal: Position = (self.size - 1, self.size - 1)
        self.max_steps = max_steps or self.size * self.size
        self.position = self.start
        self.steps = 0

    @property
    def n_states(self) -> int:
        return 16 * 9

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> int:
        self.position = self.start
        self.steps = 0
        return self.observation_index()

    def state_index(self, position: Position) -> int:
        row, col = position
        return row * self.size + col

    def observation_index(self) -> int:
        row, col = self.position
        walls = 0
        for bit, pos in enumerate(((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))):
            if not self._is_open(pos):
                walls |= 1 << bit

        goal_row, goal_col = self.goal
        row_dir = 0 if goal_row < row else 1 if goal_row == row else 2
        col_dir = 0 if goal_col < col else 1 if goal_col == col else 2
        goal_code = row_dir * 3 + col_dir
        return walls * 9 + goal_code

    def position_from_state(self, state: int) -> Position:
        return divmod(int(state), self.size)

    def step(self, action: Action) -> tuple[int, float, bool, dict]:
        row, col = self.position
        if action == 0:
            nxt = (row - 1, col)
        elif action == 1:
            nxt = (row + 1, col)
        elif action == 2:
            nxt = (row, col - 1)
        elif action == 3:
            nxt = (row, col + 1)
        else:
            raise ValueError(f"unknown action: {action}")

        blocked = not self._is_open(nxt)
        if not blocked:
            self.position = nxt

        self.steps += 1
        done = self.position == self.goal or self.steps >= self.max_steps
        reward = -0.01
        if blocked:
            reward -= 0.04
        if self.position == self.goal:
            reward += 1.0

        return self.observation_index(), reward, done, {
            "success": self.position == self.goal,
            "blocked": blocked,
            "steps": self.steps,
            "position": self.position,
        }

    def _is_open(self, position: Position) -> bool:
        row, col = position
        if row < 0 or col < 0 or row >= self.size or col >= self.size:
            return False
        return self.grid[row, col] == 0


def generate_maze(rng: np.random.Generator, spec: MazeSpec) -> np.ndarray:
    while True:
        grid = (rng.random((spec.size, spec.size)) < spec.obstacle_prob).astype(np.int8)
        grid[0, 0] = 0
        grid[spec.size - 1, spec.size - 1] = 0
        if _has_path(grid):
            return grid


def generate_mazes(count: int, seed: int, spec: MazeSpec) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [generate_maze(rng, spec) for _ in range(count)]


def _has_path(grid: np.ndarray) -> bool:
    size = grid.shape[0]
    start: Position = (0, 0)
    goal = (size - 1, size - 1)
    queue: list[Position] = [start]
    seen: set[Position] = {start}
    while queue:
        row, col = queue.pop(0)
        if (row, col) == goal:
            return True
        for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            nr, nc = nxt
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def shortest_path_length(grid: np.ndarray) -> int | None:
    size = grid.shape[0]
    start: Position = (0, 0)
    goal = (size - 1, size - 1)
    queue: list[tuple[Position, int]] = [(start, 0)]
    seen: set[Position] = {start}
    while queue:
        (row, col), dist = queue.pop(0)
        if (row, col) == goal:
            return dist
        for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            nr, nc = nxt
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def trajectory_cells(states: Iterable[int], size: int) -> frozenset[Position]:
    return frozenset(divmod(int(state), size) for state in states)
