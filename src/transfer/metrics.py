from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptationPoint:
    step: int
    success_rate: float
    avg_steps: float
    score: float


@dataclass(frozen=True)
class TransferReport:
    method: str
    source_env: str
    target_env: str
    seed: int
    zero_shot_score: float
    adaptation_auc: float
    time_to_first_success: int
    time_to_threshold: int
    final_score: float
    transfer_lift: float = 0.0
    negative_transfer: bool = False


def adaptation_auc(points: list[AdaptationPoint]) -> float:
    if not points:
        return 0.0
    if len(points) == 1:
        return float(points[0].score)

    ordered = sorted(points, key=lambda point: point.step)
    x = np.array([point.step for point in ordered], dtype=np.float64)
    y = np.array([point.score for point in ordered], dtype=np.float64)
    span = max(1.0, float(x[-1] - x[0]))
    return float(np.trapezoid(y, x) / span)


def time_to_first_success(points: list[AdaptationPoint]) -> int:
    for point in sorted(points, key=lambda item: item.step):
        if point.success_rate > 0.0:
            return point.step
    return -1


def time_to_threshold(points: list[AdaptationPoint], threshold: float) -> int:
    for point in sorted(points, key=lambda item: item.step):
        if point.score >= threshold:
            return point.step
    return -1
