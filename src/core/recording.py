from __future__ import annotations

from core.budget import MethodBudget


def annotate_method_rows(rows: list[dict], budget: MethodBudget) -> tuple[list[dict], float]:
    runtime_seconds = budget.elapsed()
    time_limited = int(budget.time_limited())
    for row in rows:
        row["runtime_seconds"] = round(runtime_seconds, 4)
        row["time_limited"] = time_limited
    return rows, runtime_seconds
