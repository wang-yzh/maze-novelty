from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents import Rollout


@dataclass(frozen=True)
class RolloutDiagnostics:
    progress_ratio: float
    max_distance_from_start: int
    final_distance_from_start: int
    revisit_ratio: float
    mobility: float
    region_transition_count: int
    new_region_count: int
    compressed_region_sequence: tuple[tuple[int, int], ...]
    doorway_like_crossings: int
    subgoal_score: float


@dataclass(frozen=True)
class MiniGridEvalDiagnostics:
    avg_subgoal_score: float
    best_subgoal_score: float
    avg_region_transitions: float
    best_region_transitions: int
    avg_new_regions: float
    best_new_regions: int
    avg_revisit_ratio: float
    avg_mobility: float
    best_max_distance_from_start: int


def analyze_rollout(rollout: Rollout, max_steps: int, region_size: int = 4) -> RolloutDiagnostics:
    positions = rollout.positions
    if not positions:
        return RolloutDiagnostics(0.0, 0, 0, 1.0, 0.0, 0, 0, (), 0, 0.0)

    start = positions[0]
    distances = [_manhattan(start, position) for position in positions]
    max_distance = max(distances)
    final_distance = distances[-1]
    progress_ratio = min(1.0, max_distance / max(1, int(max_steps**0.5) * 2))
    revisit_ratio = 1.0 - len(set(positions)) / len(positions)
    mobility = _mobility(positions)
    compressed_regions = _compressed_regions(positions, region_size)
    region_transitions = max(0, len(compressed_regions) - 1)
    new_regions = len(set(compressed_regions))
    doorway_like_crossings = _doorway_like_crossings(positions, region_size)

    new_region_norm = min(1.0, new_regions / 6.0)
    transition_norm = min(1.0, region_transitions / 8.0)
    subgoal_score = (
        0.30 * progress_ratio
        + 0.25 * new_region_norm
        + 0.20 * transition_norm
        + 0.15 * mobility
        - 0.10 * revisit_ratio
    )
    subgoal_score = max(0.0, min(1.0, subgoal_score))

    return RolloutDiagnostics(
        progress_ratio=progress_ratio,
        max_distance_from_start=max_distance,
        final_distance_from_start=final_distance,
        revisit_ratio=revisit_ratio,
        mobility=mobility,
        region_transition_count=region_transitions,
        new_region_count=new_regions,
        compressed_region_sequence=compressed_regions,
        doorway_like_crossings=doorway_like_crossings,
        subgoal_score=subgoal_score,
    )


def summarize_rollouts(rollouts: list[Rollout], max_steps: int, region_size: int = 4) -> MiniGridEvalDiagnostics:
    diagnostics = [analyze_rollout(rollout, max_steps, region_size) for rollout in rollouts]
    if not diagnostics:
        return MiniGridEvalDiagnostics(0.0, 0.0, 0.0, 0, 0.0, 0, 1.0, 0.0, 0)

    subgoals = [diag.subgoal_score for diag in diagnostics]
    transitions = [diag.region_transition_count for diag in diagnostics]
    new_regions = [diag.new_region_count for diag in diagnostics]
    revisit = [diag.revisit_ratio for diag in diagnostics]
    mobility = [diag.mobility for diag in diagnostics]
    max_distances = [diag.max_distance_from_start for diag in diagnostics]

    return MiniGridEvalDiagnostics(
        avg_subgoal_score=float(np.mean(subgoals)),
        best_subgoal_score=float(max(subgoals)),
        avg_region_transitions=float(np.mean(transitions)),
        best_region_transitions=int(max(transitions)),
        avg_new_regions=float(np.mean(new_regions)),
        best_new_regions=int(max(new_regions)),
        avg_revisit_ratio=float(np.mean(revisit)),
        avg_mobility=float(np.mean(mobility)),
        best_max_distance_from_start=int(max(max_distances)),
    )


def _compressed_regions(positions: list[tuple[int, int]], region_size: int) -> tuple[tuple[int, int], ...]:
    compressed: list[tuple[int, int]] = []
    previous = None
    for row, col in positions:
        region = (row // region_size, col // region_size)
        if region != previous:
            compressed.append(region)
            previous = region
    return tuple(compressed)


def _doorway_like_crossings(positions: list[tuple[int, int]], region_size: int) -> int:
    if len(positions) < 2:
        return 0
    crossings = 0
    for current, nxt in zip(positions, positions[1:], strict=False):
        current_region = (current[0] // region_size, current[1] // region_size)
        next_region = (nxt[0] // region_size, nxt[1] // region_size)
        if current_region != next_region:
            crossings += 1
    return crossings


def _mobility(positions: list[tuple[int, int]]) -> float:
    if len(positions) < 2:
        return 0.0
    moved = sum(next_position != position for position, next_position in zip(positions, positions[1:], strict=False))
    return moved / max(1, len(positions) - 1)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
