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
    assert config["execution_mode"] == "default"
    assert config["continuation_rule"] == "signature"
    assert config["stall_tolerance"] == 0
    assert config["min_execution_support"] == 1
    assert config["allow_trace_priors"] is True
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
    assert operate["execution_mode"] == "default"
    assert operate["continuation_rule"] == "signature"
    assert motif["abort_on_mismatch"] is True
    assert motif["mismatch_tolerance"] == 1
    assert motif["execution_mode"] == "motif_fragments"
    assert motif["continuation_rule"] == "motif_consistency"
    assert motif["stall_tolerance"] == 0
    assert motif["min_execution_support"] == 2
    assert motif["allow_trace_priors"] is False
    assert subgoal["abort_on_mismatch"] is True
    assert subgoal["mismatch_tolerance"] == 1
    assert subgoal["execution_mode"] == "default"
    assert subgoal["continuation_rule"] == "progress_guard"
    assert subgoal["stall_tolerance"] == 1
