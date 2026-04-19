from __future__ import annotations
# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import QAgent
from pretraining.artifacts import PretrainArtifact
from transfer.metrics import AdaptationPoint, adaptation_auc, time_to_first_success, time_to_threshold


def test_clone_copies_q_table_without_sharing_rng() -> None:
    rng = np.random.default_rng(7)
    agent = QAgent(4, 3, rng)
    agent.q[2, 1] = 1.5

    clone = agent.clone()

    assert clone is not agent
    assert clone.rng is not agent.rng
    np.testing.assert_array_equal(clone.q, agent.q)

    clone.q[2, 1] = -3.0
    assert agent.q[2, 1] == 1.5


def test_clone_with_seed_makes_tie_breaking_reproducible() -> None:
    agent = QAgent(2, 4, np.random.default_rng(11))

    first = agent.clone_with_seed(123)
    second = agent.clone_with_seed(123)

    first_actions = [first.act(0, epsilon=0.0) for _ in range(8)]
    second_actions = [second.act(0, epsilon=0.0) for _ in range(8)]
    assert first_actions == second_actions


def test_pretrain_artifact_best_agent_is_deterministic() -> None:
    agent = QAgent(2, 4, np.random.default_rng(99))
    artifact = PretrainArtifact(
        method="unit",
        source_env="source",
        seed=1000,
        population=[],
        hall_of_fame=[agent],
    )

    first = artifact.best_agent()
    second = artifact.best_agent()

    assert [first.act(0, 0.0) for _ in range(8)] == [second.act(0, 0.0) for _ in range(8)]


def test_transfer_metric_helpers() -> None:
    points = [
        AdaptationPoint(step=0, success_rate=0.0, avg_steps=100.0, score=0.0),
        AdaptationPoint(step=10, success_rate=0.0, avg_steps=100.0, score=0.2),
        AdaptationPoint(step=20, success_rate=0.5, avg_steps=50.0, score=0.4),
    ]

    assert adaptation_auc(points) == 0.2
    assert time_to_first_success(points) == 20
    assert time_to_threshold(points, 0.3) == 20
    assert time_to_threshold(points, 0.5) == -1
