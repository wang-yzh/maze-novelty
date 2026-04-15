from __future__ import annotations

from collections import Counter

import numpy as np

class NoveltyArchive:
    def __init__(self, size: int, max_items: int = 300):
        self.size = size
        self.max_items = max_items
        self.trajectories: list[frozenset[tuple[int, int]]] = []
        self.success_trajectories: list[frozenset[tuple[int, int]]] = []
        self.visit_counts: Counter[int] = Counter()

    def add(self, positions: list[tuple[int, int]], success: bool = False) -> None:
        cells = frozenset(positions)
        if not cells:
            return
        self.trajectories.append(cells)
        if success:
            self.success_trajectories.append(cells)
        for row, col in positions:
            self.visit_counts[row * self.size + col] += 1
        if len(self.trajectories) > self.max_items:
            self.trajectories = self.trajectories[-self.max_items :]
        if len(self.success_trajectories) > self.max_items:
            self.success_trajectories = self.success_trajectories[-self.max_items :]

    def trajectory_novelty(self, positions: list[tuple[int, int]]) -> float:
        cells = frozenset(positions)
        if not self.trajectories:
            return 1.0
        distances = [1.0 - _jaccard(cells, old) for old in self.trajectories[-80:]]
        distances.sort(reverse=True)
        k = min(5, len(distances))
        return float(np.mean(distances[:k]))

    def position_bonus(self, position: tuple[int, int]) -> float:
        row, col = position
        visits = self.visit_counts.get(row * self.size + col, 0)
        return 1.0 / np.sqrt(1.0 + visits)

    def coverage(self) -> float:
        return len(self.visit_counts) / float(self.size * self.size)

    def unique_ratio(self) -> float:
        if not self.trajectories:
            return 0.0
        return len(set(self.trajectories)) / len(self.trajectories)

    def success_diversity(self) -> float:
        if not self.success_trajectories:
            return 0.0
        return len(set(self.success_trajectories)) / len(self.success_trajectories)


def _jaccard(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))
