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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple MiniGrid diagnostic policies.")
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
        rollouts = [
            _run_policy_episode(args, env, policy, rng, episode)
            for episode in range(args.episodes)
        ]
        env.close()
        diagnostics = summarize_rollouts(rollouts, args.max_steps)
        successes = [rollout for rollout in rollouts if rollout.success]
        success_rate = len(successes) / len(rollouts)
        avg_steps = float(np.mean([rollout.steps for rollout in successes])) if successes else float(args.max_steps)
        row = {
            "policy": name,
            "success_rate": round(success_rate, 4),
            "avg_steps": round(avg_steps, 4),
            "avg_subgoal_score": round(diagnostics.avg_subgoal_score, 4),
            "best_subgoal_score": round(diagnostics.best_subgoal_score, 4),
            "avg_region_transitions": round(diagnostics.avg_region_transitions, 4),
            "best_region_transitions": diagnostics.best_region_transitions,
            "avg_new_regions": round(diagnostics.avg_new_regions, 4),
            "best_new_regions": diagnostics.best_new_regions,
            "avg_revisit_ratio": round(diagnostics.avg_revisit_ratio, 4),
            "avg_mobility": round(diagnostics.avg_mobility, 4),
            "best_max_distance_from_start": diagnostics.best_max_distance_from_start,
        }
        rows.append(row)
        print(
            name,
            "success=",
            row["success_rate"],
            "subgoal=",
            row["avg_subgoal_score"],
            "regions=",
            row["avg_new_regions"],
            "transitions=",
            row["avg_region_transitions"],
            "mobility=",
            row["avg_mobility"],
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _run_policy_episode(args, env: MiniGridTabularEnv, policy, rng: np.random.Generator, episode: int) -> Rollout:
    env.episode_seed = args.seed + 10000 + episode
    state = env.reset()
    states = [state]
    positions = [env.position]
    state_signatures = [env.state_signature]
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
        state_signatures.append(info["state_signature"])
        previous_action = action
        previous_position = positions[-2]
        state = next_state
        success = bool(info["success"])
        if done:
            break
    return Rollout(
        states,
        positions,
        actions,
        rewards,
        success,
        len(actions),
        float(np.sum(rewards)),
        state_signatures=state_signatures,
    )


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
