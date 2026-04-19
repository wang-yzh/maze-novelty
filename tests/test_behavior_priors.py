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
    PriorKindStats,
    PriorExecutionStats,
    TargetReuseConfig,
    TargetReuseItem,
    _accumulate_episode_diagnostics,
    _accumulate_execution_stats,
    _build_prior_library,
    _run_episode,
    _select_prior_for_execution,
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
    assert merged.effect_trace == high_score.effect_trace


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


def test_motif_fragment_library_filters_trace_priors_and_low_support() -> None:
    motif_prior = BehaviorPrior(
        kind="forward_run",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.70,
        support=2,
        state_trace=(0, 1, 2),
        signature_trace=(_signature(direction=0, last_action=3), _signature(direction=0, last_action=2), _signature(direction=0, last_action=2)),
    )
    low_support_prior = BehaviorPrior(
        kind="region_transition",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(1, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.80,
        support=1,
        state_trace=(3, 4, 5),
        signature_trace=(_signature(direction=0, last_action=3), _signature(direction=0, last_action=1), _signature(direction=0, last_action=2)),
    )
    trace_prior = BehaviorPrior(
        kind="target_trace",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.95,
        support=5,
        state_trace=(6, 7, 8),
        signature_trace=(_signature(direction=0, last_action=3), _signature(direction=0, last_action=2), _signature(direction=0, last_action=2)),
    )
    config = TargetReuseConfig(
        execution_mode="motif_fragments",
        min_execution_support=2,
        allow_trace_priors=False,
        match_mode=DIRECTION_AGNOSTIC_MATCH,
    )

    library = _build_prior_library(
        [
            TargetReuseItem(motif_prior),
            TargetReuseItem(low_support_prior),
            TargetReuseItem(trace_prior),
        ],
        config,
    )

    assert library is not None
    assert len(library.priors) == 1
    assert library.priors[0].kind == "forward_run"


def test_motif_fragment_library_falls_back_when_high_support_fragments_are_absent() -> None:
    low_support_motif = BehaviorPrior(
        kind="forward_run",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.70,
        support=1,
        state_trace=(0, 1, 2),
        signature_trace=(_signature(direction=0, last_action=3), _signature(direction=0, last_action=2), _signature(direction=0, last_action=2)),
    )
    config = TargetReuseConfig(
        execution_mode="motif_fragments",
        min_execution_support=2,
        allow_trace_priors=False,
        match_mode=DIRECTION_AGNOSTIC_MATCH,
    )

    library = _build_prior_library([TargetReuseItem(low_support_motif)], config)

    assert library is not None
    assert len(library.priors) == 1
    assert library.priors[0].kind == "forward_run"


def test_semantic_intent_library_falls_back_when_high_support_fragments_are_absent() -> None:
    low_support_motif = BehaviorPrior(
        kind="forward_run",
        initiation=_signature(direction=0, last_action=3),
        action_trace=(2, 2),
        termination=_signature(direction=0, last_action=2),
        score=0.70,
        support=1,
        state_trace=(0, 1, 2),
        signature_trace=(_signature(direction=0, last_action=3), _signature(direction=0, last_action=2), _signature(direction=0, last_action=2)),
    )
    config = TargetReuseConfig(
        execution_mode="semantic_intents",
        min_execution_support=2,
        allow_trace_priors=False,
        match_mode=DIRECTION_AGNOSTIC_MATCH,
    )

    library = _build_prior_library([TargetReuseItem(low_support_motif)], config)

    assert library is not None
    assert len(library.priors) == 1
    assert library.priors[0].kind == "forward_run"


def test_motif_fragment_selection_prefers_supported_shorter_prior() -> None:
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    query = _signature(direction=1, last_action=3)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=_signature(direction=0, last_action=3),
            action_trace=(2, 2, 2, 2),
            termination=_signature(direction=0, last_action=2),
            score=0.95,
            support=1,
            state_trace=(0, 1, 2, 3, 4),
            signature_trace=(
                _signature(direction=0, last_action=3),
                _signature(direction=0, last_action=2),
                _signature(direction=0, last_action=2),
                _signature(direction=0, last_action=2),
                _signature(direction=0, last_action=2),
            ),
        )
    )
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=_signature(direction=0, last_action=3),
            action_trace=(2, 2),
            termination=_signature(direction=0, last_action=2),
            score=0.80,
            support=3,
            state_trace=(5, 6, 7),
            signature_trace=(
                _signature(direction=0, last_action=3),
                _signature(direction=0, last_action=2),
                _signature(direction=0, last_action=2),
            ),
        )
    )

    selected = _select_prior_for_execution(library, query, "motif_fragments", "none")

    assert selected is not None
    assert selected.action_trace == (2, 2)
    assert selected.support == 3


def test_first_step_effect_matching_filters_incompatible_turn_prior() -> None:
    query = _signature(
        direction=0,
        last_action=3,
        local_shape=(0, 0, 1, 0, 1),
    )
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="wall_follow_left",
            initiation=query,
            action_trace=(0,),
            termination=_signature(local_shape=(1, 0, 1, 0, 1), last_action=0),
            score=0.95,
            state_trace=(0, 1),
            signature_trace=(
                query,
                _signature(local_shape=(1, 0, 1, 0, 1), last_action=0),
            ),
            effect_trace=("turn_left",),
        )
    )
    library.add(
        BehaviorPrior(
            kind="wall_follow_left",
            initiation=query,
            action_trace=(0,),
            termination=_signature(local_shape=(0, 1, 0, 1, 0), last_action=0),
            score=0.70,
            state_trace=(2, 3),
            signature_trace=(
                query,
                _signature(local_shape=(0, 1, 0, 1, 0), last_action=0),
            ),
            effect_trace=("turn_left",),
        )
    )

    loose_selected = _select_prior_for_execution(library, query, "default", "none")
    filtered_selected = _select_prior_for_execution(library, query, "default", "first_step")

    assert loose_selected is not None
    assert loose_selected.score == 0.95
    assert filtered_selected is not None
    assert filtered_selected.score == 0.70


def test_first_step_effect_matching_filters_blocked_forward_prior() -> None:
    query = _signature(
        direction=0,
        last_action=3,
        local_shape=(0, 0, 1, 0, 1),
    )
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=query,
            action_trace=(2,),
            termination=_signature(last_action=2),
            score=0.95,
            state_trace=(0, 1),
            signature_trace=(query, _signature(last_action=2)),
            effect_trace=("forward_blocked",),
        )
    )
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=query,
            action_trace=(2,),
            termination=_signature(last_action=2, topology=3),
            score=0.70,
            state_trace=(2, 3),
            signature_trace=(query, _signature(last_action=2, topology=3)),
            effect_trace=("forward_move",),
        )
    )

    selected = _select_prior_for_execution(library, query, "default", "first_step")

    assert selected is not None
    assert selected.effect_trace == ("forward_move",)


def test_semantic_intent_selection_ignores_trace_first_step_effect_filter() -> None:
    query = _signature(
        direction=0,
        last_action=3,
        local_shape=(0, 0, 1, 0, 1),
    )
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    stale_trace_prior = BehaviorPrior(
        kind="forward_run",
        initiation=query,
        action_trace=(2,),
        termination=_signature(last_action=2),
        score=0.95,
        state_trace=(0, 1),
        signature_trace=(query, _signature(last_action=2)),
        effect_trace=("forward_blocked",),
    )
    library.add(stale_trace_prior)

    default_selected = _select_prior_for_execution(library, query, "default", "first_step")
    semantic_selected = _select_prior_for_execution(library, query, "semantic_intents", "first_step")

    assert default_selected is None
    assert semantic_selected is stale_trace_prior


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
    assert priors[0].effect_trace == ("forward_move", "forward_move", "forward_move")


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
            signature_trace=tuple(signatures),
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
        abort_on_mismatch=True,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert rollout.state_signatures == signatures
    assert execution_stats.matched_prior_count >= 1
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 2
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.aborted_prior_count == 0
    kind_stats = execution_stats.kind_stats["wall_follow_right"]
    assert kind_stats.matched_prior_count >= 1
    assert kind_stats.executed_prior_count == 1
    assert kind_stats.executed_prior_steps == 2
    assert kind_stats.completed_prior_count == 1
    assert kind_stats.aborted_prior_count == 0


def test_semantic_forward_run_turns_away_from_blocked_front() -> None:
    signatures = [
        _signature(direction=0, last_action=3, local_shape=(1, 1, 0, 1, 0)),
        _signature(direction=1, last_action=1, local_shape=(0, 0, 1, 0, 1)),
        _signature(direction=1, last_action=2, local_shape=(0, 0, 1, 0, 1)),
    ]
    env = _DummyEnv(signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(11))
    agent.q[:, 0] = 5.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=signatures[0],
            action_trace=(2, 2),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(signatures),
            effect_trace=("forward_move", "forward_move"),
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
        execution_mode="semantic_intents",
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="kind_structural_guard",
        structural_patience=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.effect_abort_count == 0
    assert execution_stats.kind_stats["forward_run"].structural_evidence_steps == 1


def test_semantic_region_transition_moves_forward_when_front_is_open() -> None:
    signatures = [
        _signature(direction=0, last_action=3, local_shape=(0, 1, 1, 1, 1)),
        _signature(direction=0, last_action=2, local_shape=(0, 1, 1, 1, 1)),
        _signature(direction=0, last_action=2, local_shape=(0, 1, 1, 1, 1)),
    ]
    env = _DummyEnv(signatures, positions=[(0, 3), (0, 4), (0, 5)])
    agent = QAgent(4, 3, np.random.default_rng(11))
    agent.q[:, 0] = 5.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="region_transition",
            initiation=signatures[0],
            action_trace=(0, 0),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(signatures),
            effect_trace=("turn_left", "turn_left"),
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
        execution_mode="semantic_intents",
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="kind_structural_guard",
        structural_patience=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [2, 2]
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.kind_stats["region_transition"].structural_evidence_steps == 2


def test_run_episode_aborts_prior_when_signature_mismatches() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1),
        _signature(direction=1, last_action=2),
        _signature(direction=1, last_action=0),
    ]
    prior_signatures = [
        env_signatures[0],
        env_signatures[1],
        _signature(direction=0, last_action=2),
        _signature(direction=0, last_action=0),
    ]
    env = _DummyEnv(env_signatures)
    agent = QAgent(5, 3, np.random.default_rng(21))
    agent.q[:, 0] = 4.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=prior_signatures[0],
            action_trace=(1, 2, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(prior_signatures),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert rollout.actions[2] == 0
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 2
    assert execution_stats.aborted_prior_count == 1
    assert execution_stats.aborted_prior_steps == 1
    assert execution_stats.mismatch_abort_count == 1
    assert execution_stats.first_step_mismatch_count == 0
    assert execution_stats.completed_prior_count == 0
    kind_stats = execution_stats.kind_stats["forward_run"]
    assert kind_stats.executed_prior_count == 1
    assert kind_stats.executed_prior_steps == 2
    assert kind_stats.aborted_prior_count == 1
    assert kind_stats.aborted_prior_steps == 1
    assert kind_stats.mismatch_abort_count == 1
    assert kind_stats.completed_prior_count == 0


def test_run_episode_tolerates_single_mismatch_when_tolerance_is_one() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1, topology=3),
        _signature(direction=0, last_action=2),
        _signature(direction=0, last_action=0),
    ]
    prior_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1),
        _signature(direction=0, last_action=2),
        _signature(direction=0, last_action=0),
    ]
    env = _DummyEnv(env_signatures)
    agent = QAgent(5, 3, np.random.default_rng(21))
    agent.q[:, 0] = 4.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=prior_signatures[0],
            action_trace=(1, 2, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(prior_signatures),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        mismatch_tolerance=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:3] == [1, 2, 2]
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 3
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.first_step_mismatch_count == 1
    assert execution_stats.aborted_prior_count == 0
    assert execution_stats.aborted_prior_steps == 0


def test_run_episode_aborts_after_consecutive_mismatches_exceed_tolerance() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1, topology=3),
        _signature(direction=0, last_action=2, topology=3),
        _signature(direction=0, last_action=0),
    ]
    prior_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1),
        _signature(direction=0, last_action=2),
        _signature(direction=0, last_action=0),
    ]
    env = _DummyEnv(env_signatures)
    agent = QAgent(5, 3, np.random.default_rng(21))
    agent.q[:, 0] = 4.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=prior_signatures[0],
            action_trace=(1, 2, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(prior_signatures),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        mismatch_tolerance=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert rollout.actions[2] == 0
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 2
    assert execution_stats.aborted_prior_count == 1
    assert execution_stats.aborted_prior_steps == 1
    assert execution_stats.mismatch_abort_count == 1
    assert execution_stats.first_step_mismatch_count == 1


def test_motif_consistency_continuation_aborts_when_prior_stalls() -> None:
    signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=1, last_action=2),
        _signature(direction=1, last_action=1),
    ]
    env = _DummyEnv(signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=signatures[0],
            action_trace=(2, 1),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(signatures),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="motif_consistency",
        stall_tolerance=0,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [2, 0]
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 1
    assert execution_stats.stalled_prior_steps == 1
    assert execution_stats.first_step_stall_count == 1
    assert execution_stats.aborted_prior_count == 1
    assert execution_stats.stall_abort_count == 1
    assert execution_stats.aborted_prior_steps == 1


def test_motif_consistency_does_not_treat_turning_in_place_as_stall() -> None:
    signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=1, last_action=1),
        _signature(direction=1, last_action=2),
    ]
    env = _DummyEnv(signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="wall_follow_right",
            initiation=signatures[0],
            action_trace=(1, 2),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(signatures),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="motif_consistency",
        stall_tolerance=0,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert execution_stats.executed_prior_steps == 2
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.stalled_prior_steps == 0
    assert execution_stats.aborted_prior_count == 0


def test_progress_guard_continuation_tolerates_mismatch_while_progress_evidence_remains() -> None:
    prior_signatures = [
        _signature(direction=0, last_action=3, topology=2),
        _signature(direction=0, last_action=1, topology=2),
        _signature(direction=0, last_action=2, topology=2),
        _signature(direction=0, last_action=0, topology=2),
    ]
    env_signatures = [
        prior_signatures[0],
        _signature(direction=0, last_action=1, topology=3),
        _signature(direction=0, last_action=2, topology=3),
        _signature(direction=0, last_action=0, topology=3),
    ]
    env = _DummyEnv(env_signatures)
    agent = QAgent(5, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="region_transition",
            initiation=prior_signatures[0],
            action_trace=(1, 2, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(prior_signatures),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        mismatch_tolerance=0,
        continuation_rule="progress_guard",
        stall_tolerance=0,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:3] == [1, 2, 2]
    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 3
    assert execution_stats.mismatched_prior_steps == 3
    assert execution_stats.first_step_mismatch_count == 1
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.aborted_prior_count == 0


def test_effect_consistency_tolerates_signature_mismatch_when_effect_matches() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1, topology=3),
        _signature(direction=0, last_action=2, topology=3),
    ]
    prior_signatures = [
        env_signatures[0],
        _signature(direction=0, last_action=1, topology=2),
        _signature(direction=0, last_action=2, topology=2),
    ]
    env = _DummyEnv(env_signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="wall_follow_right",
            initiation=prior_signatures[0],
            action_trace=(1, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(prior_signatures),
            effect_trace=("turn_right", "forward_move"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        mismatch_tolerance=0,
        continuation_rule="effect_consistency",
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert execution_stats.executed_prior_steps == 2
    assert execution_stats.mismatched_prior_steps == 2
    assert execution_stats.effect_mismatched_prior_steps == 0
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.aborted_prior_count == 0


def test_effect_consistency_aborts_when_movement_effect_mismatches() -> None:
    signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=2, local_shape=(1, 1, 1, 1, 1)),
        _signature(direction=0, last_action=1),
    ]
    env = _DummyEnv(signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=signatures[0],
            action_trace=(2, 1),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(signatures),
            effect_trace=("forward_move", "turn_right"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="effect_consistency",
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [2, 0]
    assert execution_stats.executed_prior_steps == 1
    assert execution_stats.first_step_effect_mismatch_count == 1
    assert execution_stats.effect_mismatched_prior_steps == 1
    assert execution_stats.effect_abort_count == 1
    assert execution_stats.aborted_prior_count == 1
    assert execution_stats.completed_prior_count == 0


def test_effect_structural_guard_tolerates_signature_drift_with_periodic_structure() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1, topology=3),
        _signature(direction=0, last_action=2, topology=3),
        _signature(direction=0, last_action=2, topology=3),
    ]
    prior_signatures = [
        env_signatures[0],
        _signature(direction=0, last_action=1, topology=2),
        _signature(direction=0, last_action=2, topology=2),
        _signature(direction=0, last_action=2, topology=2),
    ]
    env = _DummyEnv(env_signatures, positions=[(0, 0), (0, 0), (0, 1), (0, 2)])
    agent = QAgent(5, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="wall_follow_right",
            initiation=prior_signatures[0],
            action_trace=(1, 2, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(prior_signatures),
            effect_trace=("turn_right", "forward_move", "forward_move"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        mismatch_tolerance=0,
        continuation_rule="effect_structural_guard",
        structural_patience=2,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:3] == [1, 2, 2]
    assert execution_stats.executed_prior_steps == 3
    assert execution_stats.mismatched_prior_steps == 3
    assert execution_stats.effect_mismatched_prior_steps == 0
    assert execution_stats.structural_evidence_steps == 2
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.aborted_prior_count == 0


def test_effect_structural_guard_aborts_when_structure_does_not_arrive() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1, topology=2),
        _signature(direction=1, last_action=0, local_shape=(1, 1, 1, 1, 1)),
        _signature(direction=1, last_action=1, local_shape=(1, 1, 1, 1, 1)),
    ]
    prior_signatures = [
        env_signatures[0],
        _signature(direction=0, last_action=1, topology=2),
        _signature(direction=0, last_action=0, topology=2),
        _signature(direction=0, last_action=1, topology=2),
    ]
    env = _DummyEnv(env_signatures, positions=[(0, 0), (0, 0), (0, 0), (0, 0)])
    agent = QAgent(5, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="wall_follow_right",
            initiation=prior_signatures[0],
            action_trace=(1, 0, 1),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(prior_signatures),
            effect_trace=("turn_right", "turn_left", "turn_right"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        continuation_rule="effect_structural_guard",
        structural_patience=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:3] == [1, 0, 0]
    assert execution_stats.executed_prior_steps == 2
    assert execution_stats.effect_mismatched_prior_steps == 0
    assert execution_stats.structural_evidence_steps == 0
    assert execution_stats.structural_abort_count == 1
    assert execution_stats.aborted_prior_count == 1
    assert execution_stats.aborted_prior_steps == 1
    assert execution_stats.completed_prior_count == 0


def test_kind_structural_guard_requires_forward_run_to_move() -> None:
    signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=2, local_shape=(1, 1, 1, 1, 1)),
        _signature(direction=0, last_action=1),
    ]
    env = _DummyEnv(signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=signatures[0],
            action_trace=(2, 1),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(signatures),
            effect_trace=("forward_move", "turn_right"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="kind_structural_guard",
        structural_patience=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [2, 0]
    assert execution_stats.effect_abort_count == 1
    assert execution_stats.structural_abort_count == 0
    assert execution_stats.completed_prior_count == 0


def test_kind_structural_guard_allows_wall_follow_turn_before_movement() -> None:
    env_signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1, topology=3),
        _signature(direction=0, last_action=2, topology=3),
    ]
    prior_signatures = [
        env_signatures[0],
        _signature(direction=0, last_action=1, topology=2),
        _signature(direction=0, last_action=2, topology=2),
    ]
    env = _DummyEnv(env_signatures, positions=[(0, 0), (0, 0), (0, 1)])
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4, match_mode=DIRECTION_AGNOSTIC_MATCH)
    library.add(
        BehaviorPrior(
            kind="wall_follow_right",
            initiation=prior_signatures[0],
            action_trace=(1, 2),
            termination=prior_signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2),
            signature_trace=tuple(prior_signatures),
            effect_trace=("turn_right", "forward_move"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=2,
        abort_on_mismatch=True,
        continuation_rule="kind_structural_guard",
        structural_patience=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:2] == [1, 2]
    assert execution_stats.structural_evidence_steps == 1
    assert execution_stats.completed_prior_count == 1
    assert execution_stats.aborted_prior_count == 0


def test_kind_structural_guard_requires_region_transition_evidence() -> None:
    signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1),
        _signature(direction=0, last_action=0, local_shape=(1, 1, 1, 1, 1)),
        _signature(direction=0, last_action=1, local_shape=(1, 1, 1, 1, 1)),
    ]
    env = _DummyEnv(signatures, positions=[(0, 0), (0, 0), (0, 0), (0, 0)])
    agent = QAgent(5, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="region_transition",
            initiation=signatures[0],
            action_trace=(1, 0, 1),
            termination=signatures[-1],
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=tuple(signatures),
            effect_trace=("turn_right", "turn_left", "turn_right"),
        )
    )
    execution_stats = PriorExecutionStats()

    rollout = _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        continuation_rule="kind_structural_guard",
        structural_patience=1,
        execution_stats=execution_stats,
    )

    assert rollout.actions[:3] == [1, 0, 0]
    assert execution_stats.structural_evidence_steps == 0
    assert execution_stats.structural_abort_count == 1
    assert execution_stats.completed_prior_count == 0


def test_run_episode_records_prior_truncation_when_episode_ends_mid_option() -> None:
    signatures = [
        _signature(direction=0, last_action=3),
        _signature(direction=0, last_action=1),
    ]
    env = _DummyEnv(signatures)
    agent = QAgent(4, 3, np.random.default_rng(23))
    agent.q[:, 0] = 6.0
    library = BehaviorLibrary(max_items=4)
    library.add(
        BehaviorPrior(
            kind="forward_run",
            initiation=signatures[0],
            action_trace=(1, 2, 2),
            termination=_signature(direction=0, last_action=2),
            score=0.9,
            state_trace=(0, 1, 2, 3),
            signature_trace=(
                signatures[0],
                signatures[1],
                _signature(direction=0, last_action=2),
                _signature(direction=0, last_action=2),
            ),
        )
    )
    execution_stats = PriorExecutionStats()

    _run_episode(
        env,
        agent,
        np.random.default_rng(3),
        epsilon=0.0,
        train=False,
        prior_library=library,
        prior_execute_prob=1.0,
        execute_max_actions=3,
        abort_on_mismatch=True,
        execution_stats=execution_stats,
    )

    assert execution_stats.executed_prior_count == 1
    assert execution_stats.executed_prior_steps == 1
    assert execution_stats.completed_prior_count == 0
    assert execution_stats.truncated_prior_count == 1
    kind_stats = execution_stats.kind_stats["forward_run"]
    assert kind_stats.executed_prior_count == 1
    assert kind_stats.executed_prior_steps == 1
    assert kind_stats.completed_prior_count == 0
    assert kind_stats.truncated_prior_count == 1


def test_episode_diagnostics_separate_executed_from_idle_prior_episodes() -> None:
    aggregate = PriorExecutionStats()
    executed_episode = PriorExecutionStats(
        matched_prior_count=3,
        executed_prior_count=1,
        executed_prior_steps=2,
        kind_stats={
            "forward_run": PriorKindStats(
                matched_prior_count=3,
                executed_prior_count=1,
                executed_prior_steps=2,
                completed_prior_count=1,
            )
        },
    )
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
    assert aggregate.kind_stats["forward_run"].matched_prior_count == 3
    assert aggregate.kind_stats["forward_run"].executed_prior_count == 1
    assert aggregate.kind_stats["forward_run"].executed_prior_steps == 2
    assert aggregate.kind_stats["forward_run"].completed_prior_count == 1
    assert aggregate.executed_episode_count == 1
    assert aggregate.idle_episode_count == 1
    assert aggregate.executed_episode_region_transition_total >= aggregate.idle_episode_region_transition_total


def _signature(
    direction: int = 0,
    last_action: int = 3,
    goal_bin: int = 4,
    topology: int = 2,
    local_shape: tuple[int, int, int, int, int] = (0, 0, 0, 1, 1),
) -> StateSignature:
    return StateSignature(
        direction=direction,
        local_shape=local_shape,
        goal_bin=goal_bin,
        topology=topology,
        last_action=last_action,
    )


class _DummyEnv:
    def __init__(
        self,
        signatures: list[StateSignature],
        positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self._signatures = signatures
        self._positions = positions or [(0, index) for index in range(len(signatures))]
        self.n_actions = 3
        self._index = 0
        self.position = self._positions[0]
        self.state_signature = signatures[0]

    def reset(self) -> int:
        self._index = 0
        self.position = self._positions[0]
        self.state_signature = self._signatures[0]
        return 0

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        self._index += 1
        self.position = self._positions[self._index]
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
