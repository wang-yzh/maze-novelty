from __future__ import annotations
# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import QAgent, Rollout
from core.behavior import BehaviorLibrary, BehaviorPrior
from minigrid_encoders import DIRECTION_AGNOSTIC_MATCH, StateSignature
from transfer.evaluator import (
    PriorExecutionStats,
    TargetReuseConfig,
    _accumulate_episode_diagnostics,
    _accumulate_execution_stats,
    _run_episode,
    _target_reuse_priors,
)


def test_state_signature_matching_is_goal_and_last_action_tolerant() -> None:
    reference = _signature(direction=0, last_action=3, goal_bin=4)
    candidate = _signature(direction=1, last_action=1, goal_bin=0)

    assert not reference.matches(candidate)
    assert reference.matches(candidate, mode=DIRECTION_AGNOSTIC_MATCH)


def test_behavior_library_merges_duplicate_priors() -> None:
    library = BehaviorLibrary(max_items=4)
    initiation = _signature(last_action=3)
    termination = _signature(last_action=2)
    low_score = BehaviorPrior(
        kind="forward_run",
        initiation=initiation,
        action_trace=(2, 2, 2),
        termination=termination,
        score=0.40,
        support=1,
        state_trace=(0, 1, 2, 3),
    )
    high_score = BehaviorPrior(
        kind="forward_run",
        initiation=initiation,
        action_trace=(2, 2, 2),
        termination=termination,
        score=0.80,
        support=1,
        state_trace=(10, 11, 12, 13),
    )

    library.add(low_score)
    library.add(high_score)

    assert len(library) == 1
    merged = library.priors[0]
    assert merged.support == 2
    assert np.isclose(merged.score, 0.60)
    assert merged.state_trace == high_score.state_trace


def test_behavior_library_merges_direction_variants_in_direction_agnostic_mode() -> None:
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    low = BehaviorPrior(
        kind="forward_run",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.50,
        support=1,
        state_trace=(0, 1, 2),
    )
    high = BehaviorPrior(
        kind="forward_run",
        initiation=_signature(direction=2, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=2, last_action=2),
        score=0.90,
        support=1,
        state_trace=(5, 6, 7),
    )

    library.add(low)
    library.add(high)

    assert len(library) == 1
    assert library.priors[0].support == 2
    assert library.priors[0].state_trace == high.state_trace


def test_behavior_library_best_match_respects_match_mode() -> None:
    prior = BehaviorPrior(
        kind="forward_run",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.70,
        support=1,
        state_trace=(0, 1, 2),
    )
    strict_library = BehaviorLibrary(max_items=4, match_mode="strict")
    agnostic_library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    strict_library.add(prior)
    agnostic_library.add(prior)

    query = _signature(direction=2, last_action=3)
    assert strict_library.best_match(query) is None
    assert agnostic_library.best_match(query) is not None


def test_target_reuse_extracts_navigation_prior_from_rollout() -> None:
    signatures = [
        _signature(last_action=3),
        _signature(last_action=2),
        _signature(last_action=2),
        _signature(last_action=2),
    ]
    rollout = Rollout(
        states=[0, 1, 2, 3],
        positions=[(0, 0), (0, 1), (0, 2), (0, 3)],
        actions=[2, 2, 2],
        rewards=[0.0, 0.0, 0.0],
        success=False,
        steps=3,
        total_reward=0.0,
        state_signatures=signatures,
    )

    priors = _target_reuse_priors(rollout, max_steps=16, config=TargetReuseConfig())

    assert priors
    assert priors[0].kind == "forward_run"
    assert priors[0].action_trace == (2, 2, 2)
    assert priors[0].state_trace == (0, 1, 2, 3)


def test_run_episode_executes_matching_behavior_prior_before_agent_policy() -> None:
    signatures = [
        _signature(last_action=3),
        _signature(last_action=1),
        _signature(last_action=2),
    ]
    env = _DummyEnv(signatures)
    agent = QAgent(4, 3, np.random.default_rng(11))
    agent.q[:, 0] = 5.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="wall_follow_right",
            initiation=signatures[0],
            action_trace=(1, 2),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(5),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert rollout.state_signatures == signatures
    assert execution_stats.matched_prior_count >= 1
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 2


def test_episode_diagnostics_separate_executed_from_idle_prior_episodes() -> None:
    aggregate = PriorExecutionStats()
    executed_episode = PriorExecutionStats(matched_prior_count=3, executed_prior_count=1, executed_prior_steps=2)
    idle_episode = PriorExecutionStats(matched_prior_count=1, executed_prior_count=0, executed_prior_steps=0)

    executed_rollout = Rollout(
        states=[0, 1, 2],
        positions=[(0, 0), (0, 1), (0, 2)],
        actions=[2, 2],
        rewards=[0.0, 0.0],
        success=False,
        steps=2,
        total_reward=0.0,
        state_signatures=[_signature(3), _signature(2), _signature(2)],
    )
    idle_rollout = Rollout(
        states=[0, 1, 2],
        positions=[(0, 0), (0, 0), (0, 1)],
        actions=[2, 1],
        rewards=[0.0, 0.0],
        success=False,
        steps=2,
        total_reward=0.0,
        state_signatures=[_signature(3), _signature(2), _signature(1)],
    )

    _accumulate_execution_stats(aggregate, executed_episode)
    _accumulate_episode_diagnostics(aggregate, executed_rollout, max_steps=16, episode=executed_episode)
    _accumulate_execution_stats(aggregate, idle_episode)
    _accumulate_episode_diagnostics(aggregate, idle_rollout, max_steps=16, episode=idle_episode)

    assert aggregate.matched_prior_count == 4
    assert aggregate.executed_prior_count == 1
    assert aggregate.executed_prior_steps == 2
    assert aggregate.executed_episode_count == 1
    assert aggregate.idle_episode_count == 1
    assert aggregate.executed_episode_region_transition_total >= aggregate.idle_episode_region_transition_total


def _signature(direction: int = 0, last_action: int = 3, goal_bin: int = 4) -> StateSignature:
    return StateSignature(
        direction=direction,
        local_shape=(0, 0, 0, 1, 1),
        goal_bin=goal_bin,
        topology=2,
        last_action=last_action,
    )


class _DummyEnv:
    def __init__(self, signatures: list[StateSignature]) -> None:
        self._signatures = signatures
        self.n_actions = 3
        self._index = 0
        self.position = (0, 0)
        self.state_signature = signatures[0]

    def reset(self) -> int:
        self._index = 0
        self.position = (0, 0)
        self.state_signature = self._signatures[0]
        return 0

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        self._index += 1
        self.position = (0, self._index)
        self.state_signature = self._signatures[self._index]
        done = self._index >= len(self._signatures) - 1
        return (
            self._index,
            0.0,
            done,
            {
                "success": False,
                "position": self.position,
                "state_signature": self.state_signature,
            },
        )
