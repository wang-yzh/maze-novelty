from __future__ import annotations
# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from core.behavior import BehaviorPrior
from minigrid_encoders import StateSignature
from run_pretraining_benefit_longitudinal import _deranged_indices, _shuffle_reuse_items
from transfer.evaluator import TargetReuseItem


def test_deranged_indices_avoids_fixed_points_when_possible() -> None:
    indices = _deranged_indices(5, np.random.default_rng(7))

    assert sorted(indices) == [0, 1, 2, 3, 4]
    assert all(idx != donor for idx, donor in enumerate(indices))


def test_shuffle_reuse_items_preserves_initiations_but_swaps_intents() -> None:
    first_signature = _signature(last_action=3)
    second_signature = _signature(last_action=1)
    first = TargetReuseItem(
        BehaviorPrior(
            kind="forward_run",
            initiation=first_signature,
            action_trace=(2, 2),
            termination=_signature(last_action=2),
            score=0.7,
            state_trace=(0, 1, 2),
            signature_trace=(first_signature, _signature(last_action=2), _signature(last_action=2)),
            effect_trace=("forward_move", "forward_move"),
        )
    )
    second = TargetReuseItem(
        BehaviorPrior(
            kind="region_transition",
            initiation=second_signature,
            action_trace=(0, 2),
            termination=_signature(last_action=2),
            score=0.8,
            state_trace=(3, 4, 5),
            signature_trace=(second_signature, _signature(last_action=0), _signature(last_action=2)),
            effect_trace=("turn_left", "forward_move"),
        )
    )

    shuffled = _shuffle_reuse_items([first, second], np.random.default_rng(11))

    assert [item.prior.initiation for item in shuffled] == [first_signature, second_signature]
    assert {item.prior.kind for item in shuffled} == {"forward_run", "region_transition"}
    assert shuffled[0].prior.kind != first.prior.kind
    assert shuffled[1].prior.kind != second.prior.kind


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
