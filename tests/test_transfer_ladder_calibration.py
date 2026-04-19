from __future__ import annotations
# ruff: noqa: E402

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_transfer_ladder_calibration import LadderDecisionConfig, classify_target


def test_classify_source_environment_as_regression() -> None:
    summary = classify_target(
        [
            _row("MiniGrid-FourRooms-v0", "scratch", "scratch", auc=0.2),
            _row("MiniGrid-FourRooms-v0", "target_reuse", "operate_replay_pretrain", auc=0.1),
        ],
        "MiniGrid-FourRooms-v0",
        LadderDecisionConfig(),
    )

    assert summary.tier == "source_like_regression"
    assert summary.source_like is True


def test_classify_all_zero_target_as_hard() -> None:
    summary = classify_target(
        [
            _row("MiniGrid-MultiRoom-N4-S5-v0", "scratch", "scratch", auc=0.0, final=0.0),
            _row(
                "MiniGrid-MultiRoom-N4-S5-v0",
                "target_reuse",
                "cyclic_subgoal_ecology_replay_pretrain",
                auc=0.0,
                final=0.0,
            ),
        ],
        "MiniGrid-FourRooms-v0",
        LadderDecisionConfig(),
    )

    assert summary.tier == "all_zero_hard"


def test_classify_scratch_dominated_before_control_check() -> None:
    summary = classify_target(
        [
            _row("MiniGrid-Dynamic-Obstacles-8x8-v0", "scratch", "scratch", auc=0.35),
            _row(
                "MiniGrid-Dynamic-Obstacles-8x8-v0",
                "target_reuse",
                "cyclic_motif_fast_replay_pretrain",
                auc=0.33,
                auc_lift=-0.02,
            ),
            _row(
                "MiniGrid-Dynamic-Obstacles-8x8-v0",
                "shuffled_prior_control",
                "cyclic_motif_fast_replay_pretrain",
                auc=0.32,
            ),
        ],
        "MiniGrid-FourRooms-v0",
        LadderDecisionConfig(),
    )

    assert summary.tier == "scratch_dominated"


def test_classify_control_confounded_signal() -> None:
    summary = classify_target(
        [
            _row("MiniGrid-SimpleCrossingS9N1-v0", "scratch", "scratch", auc=0.03),
            _row(
                "MiniGrid-SimpleCrossingS9N1-v0",
                "target_reuse",
                "cyclic_subgoal_ecology_replay_pretrain",
                auc=0.12,
                auc_lift=0.09,
            ),
            _row(
                "MiniGrid-SimpleCrossingS9N1-v0",
                "shuffled_prior_control",
                "cyclic_subgoal_ecology_replay_pretrain",
                auc=0.11,
            ),
        ],
        "MiniGrid-FourRooms-v0",
        LadderDecisionConfig(),
    )

    assert summary.tier == "control_confounded"


def test_classify_candidate_transfer_signal() -> None:
    summary = classify_target(
        [
            _row("MiniGrid-SimpleCrossingS9N2-v0", "scratch", "scratch", auc=0.02),
            _row(
                "MiniGrid-SimpleCrossingS9N2-v0",
                "target_reuse",
                "cyclic_subgoal_ecology_replay_pretrain",
                auc=0.14,
                auc_lift=0.12,
            ),
            _row(
                "MiniGrid-SimpleCrossingS9N2-v0",
                "shuffled_prior_control",
                "cyclic_subgoal_ecology_replay_pretrain",
                auc=0.07,
            ),
        ],
        "MiniGrid-FourRooms-v0",
        LadderDecisionConfig(),
    )

    assert summary.tier == "candidate_transfer_signal"


def _row(
    target: str,
    condition: str,
    source: str,
    *,
    auc: float,
    auc_lift: float = 0.0,
    final: float = 0.0,
    region_changes: float = 0.0,
) -> dict[str, str]:
    return {
        "target_env": target,
        "condition": condition,
        "source_method": source,
        "adaptation_auc_mean": str(auc),
        "adaptation_auc_lift_mean": str(auc_lift),
        "final_score_mean": str(final),
        "target_reuse_composition_region_change_count_mean": str(region_changes),
    }
