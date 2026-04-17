from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents import QAgent, Rollout
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from pretraining.artifacts import PretrainArtifact
from transfer.metrics import (
    AdaptationPoint,
    TransferReport,
    adaptation_auc,
    time_to_first_success,
    time_to_threshold,
)


@dataclass(frozen=True)
class AdaptationConfig:
    episodes: int = 60
    eval_every: int = 10
    eval_episodes: int = 6
    epsilon: float = 0.18
    threshold: float = 0.10


def pretrain_q_agent(
    spec: MiniGridSpec,
    seed: int,
    episodes: int,
    epsilon: float,
) -> PretrainArtifact:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec, episode_seed=seed)
    agent = QAgent(env.n_states, env.n_actions, rng)
    for episode in range(episodes):
        env.episode_seed = seed + episode
        _run_episode(env, agent, rng, epsilon=epsilon, train=True)
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
            "source_score": result.score,
            "source_success": result.success_rate,
        },
    )


def evaluate_transfer(
    artifact: PretrainArtifact,
    target_spec: MiniGridSpec,
    seed: int,
    config: AdaptationConfig,
    scratch_final_score: float = 0.0,
) -> tuple[TransferReport, list[AdaptationPoint]]:
    agent = artifact.best_agent()
    points = adapt_agent(target_spec, agent, seed, config)
    zero_shot = points[0].score if points else 0.0
    final_score = points[-1].score if points else 0.0
    lift = final_score - scratch_final_score
    return (
        TransferReport(
            method=artifact.method,
            source_env=artifact.source_env,
            target_env=target_spec.env_id,
            seed=seed,
            zero_shot_score=zero_shot,
            adaptation_auc=adaptation_auc(points),
            time_to_first_success=time_to_first_success(points),
            time_to_threshold=time_to_threshold(points, config.threshold),
            final_score=final_score,
            transfer_lift=lift,
            negative_transfer=lift < 0.0,
        ),
        points,
    )


def evaluate_scratch(
    target_spec: MiniGridSpec,
    seed: int,
    config: AdaptationConfig,
) -> tuple[TransferReport, list[AdaptationPoint]]:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(target_spec, episode_seed=seed)
    agent = QAgent(env.n_states, env.n_actions, rng)
    env.close()
    points = adapt_agent(target_spec, agent, seed, config)
    final_score = points[-1].score if points else 0.0
    return (
        TransferReport(
            method="scratch",
            source_env="none",
            target_env=target_spec.env_id,
            seed=seed,
            zero_shot_score=points[0].score if points else 0.0,
            adaptation_auc=adaptation_auc(points),
            time_to_first_success=time_to_first_success(points),
            time_to_threshold=time_to_threshold(points, config.threshold),
            final_score=final_score,
        ),
        points,
    )


def adapt_agent(
    spec: MiniGridSpec,
    agent: QAgent,
    seed: int,
    config: AdaptationConfig,
) -> list[AdaptationPoint]:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec, episode_seed=seed)
    points = [_adaptation_point(spec, agent, seed + 10000, 0, config.eval_episodes)]
    for episode in range(1, config.episodes + 1):
        env.episode_seed = seed + episode
        _run_episode(env, agent, rng, epsilon=config.epsilon, train=True)
        if episode % config.eval_every == 0 or episode == config.episodes:
            points.append(_adaptation_point(spec, agent, seed + 10000 + episode, episode, config.eval_episodes))
    env.close()
    return points


def evaluate_agent(spec: MiniGridSpec, agent: QAgent, seed: int, eval_episodes: int) -> AdaptationPoint:
    return _adaptation_point(spec, agent, seed, 0, eval_episodes)


def _adaptation_point(
    spec: MiniGridSpec,
    agent: QAgent,
    seed: int,
    step: int,
    eval_episodes: int,
) -> AdaptationPoint:
    env = MiniGridTabularEnv(spec)
    rollouts = []
    for idx in range(eval_episodes):
        env.episode_seed = seed + idx
        rollouts.append(_run_episode(env, agent, np.random.default_rng(seed + idx), epsilon=0.0, train=False))
    env.close()
    successes = [rollout for rollout in rollouts if rollout.success]
    success_rate = len(successes) / max(1, len(rollouts))
    avg_steps = float(np.mean([rollout.steps for rollout in successes])) if successes else float(spec.max_steps)
    speed = 1.0 - min(avg_steps, spec.max_steps) / spec.max_steps
    score = 0.55 * success_rate + 0.30 * speed
    return AdaptationPoint(step=step, success_rate=success_rate, avg_steps=avg_steps, score=score)


def _run_episode(
    env: MiniGridTabularEnv,
    agent: QAgent,
    rng: np.random.Generator,
    epsilon: float,
    train: bool,
) -> Rollout:
    state = env.reset()
    states = [state]
    positions = [env.position]
    actions = []
    rewards = []
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
