from __future__ import annotations
# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pretraining_transfer_diagnostic import _reuse_config_kwargs


def test_uniform_preset_uses_cli_values() -> None:
    args = SimpleNamespace(
        reuse_preset_mode="uniform",
        reuse_probe_episodes=6,
        reuse_abort_on_mismatch=True,
        reuse_mismatch_tolerance=2,
        reuse_reinforce_passes=5,
        reuse_reward=0.09,
    )

    config = _reuse_config_kwargs(args, "operate_replay_pretrain", "direction_agnostic")

    assert config["match_mode"] == "direction_agnostic"
    assert config["abort_on_mismatch"] is True
    assert config["mismatch_tolerance"] == 2
    assert config["reinforce_passes"] == 5
    assert config["reward"] == 0.09


def test_branch_specific_preset_selects_method_defaults() -> None:
    args = SimpleNamespace(
        reuse_preset_mode="branch_specific",
        reuse_probe_episodes=6,
        reuse_abort_on_mismatch=True,
        reuse_mismatch_tolerance=99,
        reuse_reinforce_passes=4,
        reuse_reward=0.075,
    )

    operate = _reuse_config_kwargs(args, "operate_replay_pretrain", "direction_agnostic")
    motif = _reuse_config_kwargs(args, "cyclic_motif_fast_replay_pretrain", "direction_agnostic")
    subgoal = _reuse_config_kwargs(args, "cyclic_subgoal_ecology_replay_pretrain", "direction_agnostic")

    assert operate["abort_on_mismatch"] is False
    assert operate["mismatch_tolerance"] == 0
    assert motif["abort_on_mismatch"] is True
    assert motif["mismatch_tolerance"] == 1
    assert subgoal["abort_on_mismatch"] is True
    assert subgoal["mismatch_tolerance"] == 1
