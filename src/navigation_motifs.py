from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from agents import Rollout


FORWARD_ACTION = 2
LEFT_ACTION = 0
RIGHT_ACTION = 1


@dataclass(frozen=True)
class NavigationMotif:
    kind: str
    start: int
    end: int
    actions: tuple[int, ...]
    positions: tuple[tuple[int, int], ...]
    quality: float
    mobility: float
    region_transitions: int
    revisit_reduction: float
    collision_proxy_rate: float


@dataclass(frozen=True)
class NavigationMotifSummary:
    motif_count: int
    forward_run_count: int
    wall_follow_left_count: int
    wall_follow_right_count: int
    region_transition_count: int
    unstuck_count: int
    avg_quality: float
    best_quality: float
    avg_mobility: float
    avg_collision_proxy_rate: float


def extract_navigation_motifs(
    rollout: Rollout,
    region_size: int = 4,
    max_window: int = 12,
) -> list[NavigationMotif]:
    if len(rollout.actions) < 3 or len(rollout.positions) < 4:
        return []

    motifs: list[NavigationMotif] = []
    motifs.extend(_extract_forward_runs(rollout, region_size, max_window))
    motifs.extend(_extract_wall_follow(rollout, region_size, max_window))
    motifs.extend(_extract_region_transitions(rollout, region_size, max_window))
    motifs.extend(_extract_unstuck(rollout, region_size, max_window))
    return _dedupe_motifs(motifs)


def summarize_navigation_motifs(motifs: list[NavigationMotif]) -> NavigationMotifSummary:
    if not motifs:
        return NavigationMotifSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    counts = Counter(motif.kind for motif in motifs)
    qualities = [motif.quality for motif in motifs]
    mobilities = [motif.mobility for motif in motifs]
    collision_rates = [motif.collision_proxy_rate for motif in motifs]
    return NavigationMotifSummary(
        motif_count=len(motifs),
        forward_run_count=counts["forward_run"],
        wall_follow_left_count=counts["wall_follow_left"],
        wall_follow_right_count=counts["wall_follow_right"],
        region_transition_count=counts["region_transition"],
        unstuck_count=counts["unstuck"],
        avg_quality=float(np.mean(qualities)),
        best_quality=float(max(qualities)),
        avg_mobility=float(np.mean(mobilities)),
        avg_collision_proxy_rate=float(np.mean(collision_rates)),
    )


def summarize_navigation_rollouts(rollouts: list[Rollout], region_size: int = 4) -> NavigationMotifSummary:
    motifs = []
    for rollout in rollouts:
        motifs.extend(extract_navigation_motifs(rollout, region_size))
    return summarize_navigation_motifs(motifs)


def _extract_forward_runs(rollout: Rollout, region_size: int, max_window: int) -> list[NavigationMotif]:
    motifs = []
    start = None
    for idx, action in enumerate(rollout.actions + [-1]):
        if action == FORWARD_ACTION:
            if start is None:
                start = idx
            continue
        if start is not None:
            end = idx
            if end - start >= 3:
                motif = _make_motif("forward_run", rollout, start, min(end, start + max_window), region_size)
                if motif.mobility >= 0.50 and motif.collision_proxy_rate <= 0.35:
                    motifs.append(motif)
            start = None
    return motifs


def _extract_wall_follow(rollout: Rollout, region_size: int, max_window: int) -> list[NavigationMotif]:
    motifs = []
    for idx in range(len(rollout.actions) - 2):
        action = rollout.actions[idx]
        if action != FORWARD_ACTION:
            continue
        if rollout.positions[idx + 1] != rollout.positions[idx]:
            continue
        turn = rollout.actions[idx + 1]
        after_turn = rollout.actions[idx + 2]
        if turn not in {LEFT_ACTION, RIGHT_ACTION} or after_turn != FORWARD_ACTION:
            continue
        kind = "wall_follow_left" if turn == LEFT_ACTION else "wall_follow_right"
        start = max(0, idx - 1)
        end = min(len(rollout.actions), start + max_window)
        motifs.append(_make_motif(kind, rollout, start, end, region_size))
    return motifs


def _extract_region_transitions(rollout: Rollout, region_size: int, max_window: int) -> list[NavigationMotif]:
    motifs = []
    for idx in range(len(rollout.positions) - 1):
        if _region(rollout.positions[idx], region_size) == _region(rollout.positions[idx + 1], region_size):
            continue
        start = max(0, idx - max_window // 2)
        end = min(len(rollout.actions), start + max_window)
        motifs.append(_make_motif("region_transition", rollout, start, end, region_size))
    return motifs


def _extract_unstuck(rollout: Rollout, region_size: int, max_window: int) -> list[NavigationMotif]:
    motifs = []
    window = min(max_window, 10)
    if len(rollout.actions) < window:
        return motifs
    stride = max(1, window // 2)
    for start in range(0, len(rollout.actions) - window + 1, stride):
        end = start + window
        positions = rollout.positions[start : end + 1]
        midpoint = len(positions) // 2
        before = positions[: midpoint + 1]
        after = positions[midpoint:]
        revisit_before = _revisit_ratio(before)
        revisit_after = _revisit_ratio(after)
        mobility_before = _mobility(before)
        mobility_after = _mobility(after)
        if revisit_before - revisit_after >= 0.12 or mobility_after - mobility_before >= 0.20:
            motifs.append(_make_motif("unstuck", rollout, start, end, region_size))
    return motifs


def _make_motif(kind: str, rollout: Rollout, start: int, end: int, region_size: int) -> NavigationMotif:
    end = max(start + 1, min(end, len(rollout.actions)))
    actions = tuple(rollout.actions[start:end])
    positions = tuple(rollout.positions[start : end + 1])
    mobility = _mobility(list(positions))
    region_transitions = _region_transitions(list(positions), region_size)
    revisit_reduction = _revisit_reduction(list(positions))
    collision_proxy_rate = _collision_proxy_rate(actions, positions)
    compactness = 1.0 - min(1.0, max(0, len(actions) - 4) / 12.0)
    quality = (
        0.30 * mobility
        + 0.25 * min(1.0, region_transitions)
        + 0.20 * max(0.0, revisit_reduction)
        + 0.15 * compactness
        + 0.10 * (1.0 - collision_proxy_rate)
    )
    return NavigationMotif(
        kind=kind,
        start=start,
        end=end,
        actions=actions,
        positions=positions,
        quality=max(0.0, min(1.0, quality)),
        mobility=mobility,
        region_transitions=region_transitions,
        revisit_reduction=revisit_reduction,
        collision_proxy_rate=collision_proxy_rate,
    )


def _dedupe_motifs(motifs: list[NavigationMotif]) -> list[NavigationMotif]:
    best_by_key: dict[tuple[str, int, int], NavigationMotif] = {}
    for motif in motifs:
        key = (motif.kind, motif.start, motif.end)
        previous = best_by_key.get(key)
        if previous is None or motif.quality > previous.quality:
            best_by_key[key] = motif
    return sorted(best_by_key.values(), key=lambda motif: motif.quality, reverse=True)


def _region(position: tuple[int, int], region_size: int) -> tuple[int, int]:
    row, col = position
    return row // region_size, col // region_size


def _region_transitions(positions: list[tuple[int, int]], region_size: int) -> int:
    if len(positions) < 2:
        return 0
    return sum(
        _region(position, region_size) != _region(next_position, region_size)
        for position, next_position in zip(positions, positions[1:], strict=False)
    )


def _mobility(positions: list[tuple[int, int]]) -> float:
    if len(positions) < 2:
        return 0.0
    moved = sum(position != next_position for position, next_position in zip(positions, positions[1:], strict=False))
    return moved / max(1, len(positions) - 1)


def _revisit_ratio(positions: list[tuple[int, int]]) -> float:
    if not positions:
        return 1.0
    return 1.0 - len(set(positions)) / len(positions)


def _revisit_reduction(positions: list[tuple[int, int]]) -> float:
    if len(positions) < 4:
        return 0.0
    midpoint = len(positions) // 2
    before = positions[: midpoint + 1]
    after = positions[midpoint:]
    return _revisit_ratio(before) - _revisit_ratio(after)


def _collision_proxy_rate(actions: tuple[int, ...], positions: tuple[tuple[int, int], ...]) -> float:
    forward_count = 0
    collisions = 0
    for idx, action in enumerate(actions):
        if action != FORWARD_ACTION:
            continue
        forward_count += 1
        if idx + 1 < len(positions) and positions[idx + 1] == positions[idx]:
            collisions += 1
    if forward_count == 0:
        return 0.0
    return collisions / forward_count
