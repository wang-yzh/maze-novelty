from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class MethodBudget:
    max_seconds: float | None = None
    started_at: float | None = None

    @classmethod
    def from_seconds(cls, seconds: float) -> "MethodBudget":
        return cls(max_seconds=seconds if seconds > 0 else None)

    def start(self) -> "MethodBudget":
        self.started_at = time.perf_counter()
        return self

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.perf_counter() - self.started_at

    def expired(self) -> bool:
        if self.max_seconds is None:
            return False
        return self.elapsed() >= self.max_seconds

    def time_limited(self) -> bool:
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds
