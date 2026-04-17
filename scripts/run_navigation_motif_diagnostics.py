from __future__ import annotations
# ruff: noqa: E402

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from agents import Rollout
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from minigrid_diagnostics import summarize_rollouts
from navigation_motifs import extract_navigation_motifs, summarize_navigation_motifs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract navigation motifs from simple MiniGrid policies.")
    parser.add_argument("--env-id", default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--state-encoder", choices=["compact", "geometry"], default="geometry")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    spec = MiniGridSpec(args.env_id, args.max_steps, args.seed, args.state_encoder)
    policies = {
        "random": _random_policy,
        "forward_biased_random": _forward_biased_random_policy,
        "wall_follow_left": _wall_follow_left_policy,
        "wall_follow_right": _wall_follow_right_policy,
    }

    rows = []
    for name, policy in policies.items():
        env = MiniGridTabularEnv(spec)
        rollouts = [_run_policy_episode(args, env, policy, rng, episode) for episode in range(args.episodes)]
        env.close()
        diagnostics = summarize_rollouts(rollouts, args.max_steps)
        motifs = []
        for rollout in rollouts:
            motifs.extend(extract_navigation_motifs(rollout))
        motif_summary = summarize_navigation_motifs(motifs)
        successes = [rollout for rollout in rollouts if rollout.success]
        row = {
            "policy": name,
            "success_rate": round(len(successes) / max(1, len(rollouts)), 4),
            "avg_subgoal_score": round(diagnostics.avg_subgoal_score, 4),
            "avg_region_transitions": round(diagnostics.avg_region_transitions, 4),
            "avg_mobility": round(diagnostics.avg_mobility, 4),
            "navigation_motif_count": motif_summary.motif_count,
            "forward_run_count": motif_summary.forward_run_count,
            "wall_follow_left_count": motif_summary.wall_follow_left_count,
            "wall_follow_right_count": motif_summary.wall_follow_right_count,
            "region_transition_motif_count": motif_summary.region_transition_count,
            "unstuck_motif_count": motif_summary.unstuck_count,
            "avg_navigation_quality": round(motif_summary.avg_quality, 4),
            "best_navigation_quality": round(motif_summary.best_quality, 4),
            "motif_avg_mobility": round(motif_summary.avg_mobility, 4),
            "collision_proxy_rate": round(motif_summary.avg_collision_proxy_rate, 4),
        }
        rows.append(row)
        print(
            name,
            "success=",
            row["success_rate"],
            "transitions=",
            row["avg_region_transitions"],
            "mobility=",
            row["avg_mobility"],
            "nav_motifs=",
            row["navigation_motif_count"],
            "wall_left=",
            row["wall_follow_left_count"],
            "wall_right=",
            row["wall_follow_right_count"],
            "quality=",
            row["avg_navigation_quality"],
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _run_policy_episode(args, env: MiniGridTabularEnv, policy, rng: np.random.Generator, episode: int) -> Rollout:
    env.episode_seed = args.seed + 12000 + episode
    state = env.reset()
    states = [state]
    positions = [env.position]
    actions = []
    rewards = []
    success = False
    previous_position = env.position
    previous_action = 2
    while True:
        action = policy(rng, previous_action, previous_position, env.position)
        next_state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(info["position"])
        previous_action = action
        previous_position = positions[-2]
        state = next_state
        success = bool(info["success"])
        if done:
            break
    return Rollout(states, positions, actions, rewards, success, len(actions), float(np.sum(rewards)))


def _random_policy(rng: np.random.Generator, _previous_action: int, _previous_position, _position) -> int:
    return int(rng.integers(3))


def _forward_biased_random_policy(rng: np.random.Generator, _previous_action: int, _previous_position, _position) -> int:
    return int(rng.choice([0, 1, 2], p=[0.18, 0.18, 0.64]))


def _wall_follow_left_policy(rng: np.random.Generator, previous_action: int, previous_position, position) -> int:
    if position == previous_position and previous_action == 2:
        return 0
    if previous_action == 0:
        return 2
    if rng.random() < 0.15:
        return 0
    return 2


def _wall_follow_right_policy(rng: np.random.Generator, previous_action: int, previous_position, position) -> int:
    if position == previous_position and previous_action == 2:
        return 1
    if previous_action == 1:
        return 2
    if rng.random() < 0.15:
        return 1
    return 2


if __name__ == "__main__":
    main()
