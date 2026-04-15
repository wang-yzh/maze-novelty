from __future__ import annotations

from pathlib import Path
import tomllib


DEFAULT_CONFIG = {
    "seed": 7,
    "size": 12,
    "train_mazes": 18,
    "test_mazes": 8,
    "generations": 45,
    "population": 18,
    "episodes_per_agent": 5,
    "eval_every": 3,
    "novelty_patience": 5,
    "obstacle_prob": 0.22,
    "make_plots": True,
}


def load_config(path: Path | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    config.update(data.get("experiment", {}))
    return config
