from __future__ import annotations
# ruff: noqa: E402

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache").mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict]) -> None:
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x = [row["generation"] for row in method_rows]
        axes[0].plot(x, [row["test_success"] for row in method_rows], label=method)
        axes[1].plot(x, [row["test_avg_steps"] for row in method_rows], label=method)
        axes[2].plot(x, [row["archive_coverage"] for row in method_rows], label=method)
    axes[0].set_title("Test success")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_title("Avg steps on success")
    axes[2].set_title("Archive coverage")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_paths(path: Path, grid: np.ndarray, method_rollouts: dict[str, list[tuple[int, int]]]) -> None:
    cols = len(method_rollouts)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, (method, states) in zip(axes, method_rollouts.items()):
        canvas = np.ones((*grid.shape, 3), dtype=float)
        canvas[grid == 1] = np.array([0.08, 0.08, 0.08])
        for row, col in states:
            canvas[row, col] = np.array([0.35, 0.68, 0.92])
        canvas[0, 0] = np.array([0.20, 0.75, 0.32])
        canvas[-1, -1] = np.array([0.88, 0.20, 0.22])
        ax.imshow(canvas, interpolation="nearest")
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, grid.shape[0] - 0.5)
        ax.set_ylim(grid.shape[0] - 0.5, -0.5)
        _draw_grid(ax, grid.shape[0])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_grid(ax, size: int) -> None:
    for i in range(size + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.35)
        ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.35)
