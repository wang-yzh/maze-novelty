from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents import QAgent, Rollout
from evolution import EvalResult


@dataclass
class NicheElite:
    agent: QAgent
    score: float
    descriptor: tuple[int, ...]


class NicheArchive:
    def __init__(self, max_cells: int = 512, max_per_cell: int = 2):
        self.max_cells = max_cells
        self.max_per_cell = max_per_cell
        self.cells: dict[tuple[int, ...], list[NicheElite]] = {}

    def add(self, agent: QAgent, rollout: Rollout, score: float, max_steps: int) -> None:
        descriptor = behavior_descriptor(rollout, max_steps)
        elites = self.cells.setdefault(descriptor, [])
        elites.append(NicheElite(agent.clone(), score, descriptor))
        elites.sort(key=lambda elite: elite.score, reverse=True)
        del elites[self.max_per_cell :]
        self._trim_cells()

    def sample(self, rng: np.random.Generator, count: int) -> list[QAgent]:
        elites = self._all_elites()
        if not elites:
            return []
        return [elites[int(rng.integers(len(elites)))].agent.clone() for _ in range(count)]

    def best(self, count: int) -> list[QAgent]:
        elites = sorted(self._all_elites(), key=lambda elite: elite.score, reverse=True)
        return [elite.agent.clone() for elite in elites[:count]]

    def local_survivors(self, per_cell: int = 1, limit: int | None = None) -> list[QAgent]:
        survivors: list[NicheElite] = []
        for elites in self.cells.values():
            survivors.extend(elites[:per_cell])
        survivors.sort(key=lambda elite: elite.score, reverse=True)
        if limit is not None:
            survivors = survivors[:limit]
        return [elite.agent.clone() for elite in survivors]

    def crowding_penalty(self, rollout: Rollout, max_steps: int) -> float:
        elites = self.cells.get(behavior_descriptor(rollout, max_steps), [])
        if not elites:
            return 0.0
        return float(np.log1p(len(elites)) / np.log1p(max(2, self.max_per_cell + 1)))

    def __len__(self) -> int:
        return len(self.cells)

    def _all_elites(self) -> list[NicheElite]:
        return [elite for elites in self.cells.values() for elite in elites]

    def _trim_cells(self) -> None:
        while len(self.cells) > self.max_cells:
            weakest = min(
                self.cells,
                key=lambda descriptor: max(elite.score for elite in self.cells[descriptor]),
            )
            del self.cells[weakest]


class MotifBank:
    def __init__(self, max_items: int = 128):
        self.max_items = max_items
        self.items: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def add_success(self, rollout: Rollout, window: int = 8) -> None:
        if not rollout.success or len(rollout.actions) < 2:
            return
        spans = []
        end_start = max(0, len(rollout.actions) - window)
        spans.append((end_start, len(rollout.actions)))
        if len(rollout.actions) > window:
            best_start = most_mobile_window(rollout.positions, window)
            spans.append((best_start, best_start + window))
        for start, end in spans:
            states = tuple(rollout.states[start : end + 1])
            actions = tuple(rollout.actions[start:end])
            if actions:
                self.items.append((states, actions))
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def reinforce(self, agent: QAgent, rng: np.random.Generator, passes: int = 3, reward: float = 0.07) -> None:
        if not self.items:
            return
        for _ in range(passes):
            states, actions = self.items[int(rng.integers(len(self.items)))]
            reinforce_action_trace(agent, states, actions, reward)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class ScoredMotif:
    states: tuple[int, ...]
    actions: tuple[int, ...]
    score: float
    kind: str


class ScoredMotifBank:
    def __init__(self, max_items: int = 160):
        self.max_items = max_items
        self.items: list[ScoredMotif] = []
        self.confirmed_count = 0
        self.candidate_count = 0

    def add_rollout(self, rollout: Rollout, max_steps: int, novelty: float = 0.0, window: int = 8) -> str | None:
        kind = motif_kind(rollout, max_steps)
        if kind is None or len(rollout.actions) < 2:
            return None

        base_score = motif_quality(rollout, max_steps, novelty, kind)
        spans = _motif_spans(rollout, window)
        for start, end in spans:
            states = tuple(rollout.states[start : end + 1])
            actions = tuple(rollout.actions[start:end])
            if actions:
                mobility = _segment_mobility(rollout.positions[start : end + 1])
                self.items.append(ScoredMotif(states, actions, base_score + 0.08 * mobility, kind))
        self.items.sort(key=lambda item: item.score, reverse=True)
        del self.items[self.max_items :]
        if kind.startswith("confirmed"):
            self.confirmed_count += 1
        else:
            self.candidate_count += 1
        return kind

    def reinforce(self, agent: QAgent, rng: np.random.Generator, passes: int = 3, reward: float = 0.08) -> None:
        if not self.items:
            return
        weights = np.array([max(0.01, item.score) for item in self.items], dtype=np.float64)
        weights = weights / weights.sum()
        for _ in range(passes):
            item = self.items[int(rng.choice(len(self.items), p=weights))]
            kind_reward = reward if item.kind.startswith("confirmed") else reward * 0.45
            reinforce_action_trace(agent, item.states, item.actions, kind_reward)

    def kind_count(self, prefix: str) -> int:
        return sum(item.kind.startswith(prefix) for item in self.items)

    def __len__(self) -> int:
        return len(self.items)


def motif_kind(rollout: Rollout, max_steps: int) -> str | None:
    if rollout.success:
        if rollout.steps <= fast_success_threshold(max_steps):
            return "confirmed_fast"
        if rollout.steps <= medium_success_threshold(max_steps):
            return "confirmed_medium"
        return None
    progress = rollout_progress(rollout, max_steps)
    mobility = rollout_mobility(rollout)
    if progress >= 0.28:
        return "candidate_progress"
    if mobility >= 0.55:
        return "candidate_mobility"
    return None


def motif_quality(rollout: Rollout, max_steps: int, novelty: float, kind: str) -> float:
    parent_success = 1.0 if rollout.success else 0.35
    parent_speed = 1.0 - min(rollout.steps, max_steps) / max_steps
    mobility = rollout_mobility(rollout)
    kind_bonus = 0.20 if kind == "confirmed_fast" else 0.08 if kind == "confirmed_medium" else 0.0
    return 0.35 * parent_success + 0.25 * parent_speed + 0.20 * mobility + 0.10 * novelty + kind_bonus


def fast_success_threshold(max_steps: int) -> int:
    return max(16, int(max_steps * 0.12))


def medium_success_threshold(max_steps: int) -> int:
    return max(fast_success_threshold(max_steps) + 1, int(max_steps * 0.25))


def rollout_progress(rollout: Rollout, max_steps: int) -> float:
    if len(rollout.positions) < 2:
        return 0.0
    start = rollout.positions[0]
    end = rollout.positions[-1]
    net = abs(end[0] - start[0]) + abs(end[1] - start[1])
    return min(1.0, net / max(1, int(max_steps**0.5) * 2))


def rollout_mobility(rollout: Rollout) -> float:
    if len(rollout.positions) < 2:
        return 0.0
    moved = sum(rollout.positions[idx + 1] != rollout.positions[idx] for idx in range(len(rollout.positions) - 1))
    return moved / max(1, len(rollout.positions) - 1)


@dataclass
class StressGateResult:
    survivors: list[QAgent]
    scores: list[float]
    pass_rate: float


def stress_score(result: EvalResult, max_steps: int) -> float:
    speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
    robustness = result.success_rate
    simplicity = speed
    return 0.45 * result.success_rate + 0.25 * robustness + 0.20 * speed + 0.10 * simplicity


def speciated_selection_score(
    result: EvalResult,
    stress_result: EvalResult,
    niche_quality: float,
    max_steps: int,
) -> float:
    speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
    stress = stress_score(stress_result, max_steps)
    simplicity = speed
    return (
        0.45 * result.success_rate
        + 0.25 * speed
        + 0.15 * stress
        + 0.10 * niche_quality
        + 0.05 * simplicity
    )


def motif_selection_score(result: EvalResult, motif_count: int, novelty: float, max_steps: int) -> float:
    speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
    motif_reuse = min(1.0, motif_count / 32.0)
    return 0.40 * result.success_rate + 0.30 * speed + 0.15 * motif_reuse + 0.10 * novelty + 0.05 * result.stability


def resource_selection_score(
    task_score: float,
    stress_value: float,
    novelty: float,
    underexplored_bonus: float,
    crowding_penalty: float,
) -> float:
    return 0.40 * task_score + 0.20 * stress_value + 0.15 * novelty + 0.15 * underexplored_bonus - 0.10 * crowding_penalty


def behavior_descriptor(rollout: Rollout, max_steps: int) -> tuple[int, int, int, int, int, int, int]:
    row, col = rollout.positions[-1]
    final_region = min(15, (row // 4) * 4 + (col // 4))
    success_flag = int(rollout.success)
    speed_bin = min(3, int(4 * (1.0 - min(rollout.steps, max_steps) / max_steps)))
    coverage_bin = min(3, int(4 * len(set(rollout.positions)) / max_steps))
    turn_bin = _turn_bin(rollout.actions)
    revisit_bin = _revisit_bin(rollout.positions)
    progress_bin = _progress_bin(rollout.positions)
    return final_region, success_flag, speed_bin, coverage_bin, turn_bin, revisit_bin, progress_bin


def most_mobile_window(positions: list[tuple[int, int]], window: int) -> int:
    if len(positions) <= window + 1:
        return 0
    best_start = 0
    best_distance = -1
    for start in range(0, len(positions) - window):
        segment = positions[start : start + window + 1]
        distance = sum(
            abs(segment[idx + 1][0] - segment[idx][0]) + abs(segment[idx + 1][1] - segment[idx][1])
            for idx in range(len(segment) - 1)
        )
        if distance > best_distance:
            best_distance = distance
            best_start = start
    return best_start


def _motif_spans(rollout: Rollout, window: int) -> list[tuple[int, int]]:
    spans = []
    end_start = max(0, len(rollout.actions) - window)
    spans.append((end_start, len(rollout.actions)))
    if len(rollout.actions) > window:
        best_start = most_mobile_window(rollout.positions, window)
        spans.append((best_start, best_start + window))
    return spans


def _segment_mobility(positions: list[tuple[int, int]]) -> float:
    if len(positions) < 2:
        return 0.0
    moved = sum(positions[idx + 1] != positions[idx] for idx in range(len(positions) - 1))
    return moved / max(1, len(positions) - 1)


def reinforce_action_trace(agent: QAgent, states: list[int] | tuple[int, ...], actions: list[int] | tuple[int, ...], reward: float) -> None:
    for idx, action in enumerate(actions):
        state = states[idx]
        next_state = states[idx + 1]
        done = idx == len(actions) - 1
        shaped_reward = reward + (1.0 if done else 0.0)
        agent.update(state, action, shaped_reward, next_state, done)


def _turn_bin(actions: list[int]) -> int:
    if len(actions) < 2:
        return 0
    turns = sum(int(actions[idx] != actions[idx - 1]) for idx in range(1, len(actions)))
    return min(3, int(4 * turns / max(1, len(actions) - 1)))


def _revisit_bin(positions: list[tuple[int, int]]) -> int:
    if not positions:
        return 0
    revisit_ratio = 1.0 - len(set(positions)) / len(positions)
    return min(3, int(4 * revisit_ratio))


def _progress_bin(positions: list[tuple[int, int]]) -> int:
    if len(positions) < 2:
        return 0
    start = positions[0]
    end = positions[-1]
    net = abs(end[0] - start[0]) + abs(end[1] - start[1])
    path = sum(
        abs(positions[idx + 1][0] - positions[idx][0]) + abs(positions[idx + 1][1] - positions[idx][1])
        for idx in range(len(positions) - 1)
    )
    if path <= 0:
        return 0
    return min(3, int(4 * net / path))
