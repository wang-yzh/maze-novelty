from __future__ import annotations

from collections.abc import Callable

import numpy as np

from agents import QAgent, Rollout
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from pretraining.artifacts import PretrainArtifact
from replay import SuccessReplayBank
from transfer.evaluator import evaluate_agent


PretrainBuilder = Callable[[MiniGridSpec, int, int, float], PretrainArtifact]


def pretrain_simple_q_agent(
    spec: MiniGridSpec,
    seed: int,
    episodes: int,
    epsilon: float,
) -> PretrainArtifact:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec, episode_seed=seed)
    agent = QAgent(env.n_states, env.n_actions, rng)
    successes = 0
    for episode in range(episodes):
        env.episode_seed = seed + episode
        rollout = run_minigrid_episode(env, agent, rng, epsilon=epsilon, train=True)
        successes += int(rollout.success)
    result = evaluate_agent(spec, agent, seed + 9000, eval_episodes=6)
    env.close()
    return PretrainArtifact(
        method="simple_q_pretrain",
        source_env=spec.env_id,
        seed=seed,
        population=[agent.clone()],
        hall_of_fame=[agent.clone()],
        metadata={
            "pretrain_episodes": episodes,
            "pretrain_epsilon": epsilon,
            "train_successes": successes,
            "source_score": result.score,
            "source_success": result.success_rate,
        },
    )


def pretrain_operate_replay_agent(
    spec: MiniGridSpec,
    seed: int,
    episodes: int,
    epsilon: float,
) -> PretrainArtifact:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec, episode_seed=seed)
    agent = QAgent(env.n_states, env.n_actions, rng)
    replay_bank = SuccessReplayBank(max_items=120)
    phase_counts = {"explore": 0, "operate": 0, "replay": 0}
    train_successes = 0
    replay_passes = 0

    for episode in range(episodes):
        phase = _operate_replay_phase(episode)
        phase_counts[phase] += 1
        env.episode_seed = seed + episode

        if phase == "explore":
            phase_epsilon = max(epsilon, 0.38)
        elif phase == "operate":
            phase_epsilon = min(epsilon, 0.14)
        else:
            phase_epsilon = min(epsilon, 0.10)

        rollout = run_minigrid_episode(env, agent, rng, epsilon=phase_epsilon, train=True)
        replay_bank.add(rollout)
        train_successes += int(rollout.success)

        if phase == "replay":
            replay_bank.reinforce(agent, rng, passes=4, reward=0.10)
            replay_passes += 4

    result = evaluate_agent(spec, agent, seed + 9000, eval_episodes=6)
    env.close()
    return PretrainArtifact(
        method="operate_replay_pretrain",
        source_env=spec.env_id,
        seed=seed,
        population=[agent.clone()],
        hall_of_fame=[agent.clone()],
        metadata={
            "pretrain_episodes": episodes,
            "base_epsilon": epsilon,
            "phase_counts": phase_counts,
            "train_successes": train_successes,
            "replay_bank_size": len(replay_bank),
            "replay_passes": replay_passes,
            "source_score": result.score,
            "source_success": result.success_rate,
        },
    )


def build_pretrain_artifact(
    method: str,
    spec: MiniGridSpec,
    seed: int,
    episodes: int,
    epsilon: float,
) -> PretrainArtifact:
    builders: dict[str, PretrainBuilder] = {
        "simple_q_pretrain": pretrain_simple_q_agent,
        "operate_replay_pretrain": pretrain_operate_replay_agent,
    }
    try:
        builder = builders[method]
    except KeyError as exc:
        available = ", ".join(sorted(builders))
        msg = f"Unknown pretraining method {method!r}. Available: {available}"
        raise ValueError(msg) from exc
    return builder(spec, seed, episodes, epsilon)


def run_minigrid_episode(
    env: MiniGridTabularEnv,
    agent: QAgent,
    rng: np.random.Generator,
    epsilon: float,
    train: bool,
) -> Rollout:
    state = env.reset()
    states = [state]
    positions = [env.position]
    actions: list[int] = []
    rewards: list[float] = []
    success = False

    while True:
        if rng.random() < epsilon:
            action = int(rng.integers(env.n_actions))
        else:
            action = agent.act(state, 0.0)
        next_state, reward, done, info = env.step(action)
        if train:
            agent.update(state, action, reward, next_state, done)
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(info["position"])
        state = next_state
        success = bool(info["success"])
        if done:
            break

    return Rollout(states, positions, actions, rewards, success, len(actions), float(np.sum(rewards)))


def _operate_replay_phase(episode: int) -> str:
    slot = episode % 10
    if slot < 4:
        return "explore"
    if slot < 8:
        return "operate"
    return "replay"
