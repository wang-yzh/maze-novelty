from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents import QAgent, Rollout
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from minigrid_diagnostics import analyze_rollout
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


@dataclass(frozen=True)
class TargetReuseConfig:
    probe_episodes: int = 6
    probe_epsilon: float = 0.03
    min_subgoal_score: float = 0.018
    min_mobility: float = 0.035
    reinforce_passes: int = 4
    reward: float = 0.075
    early_fraction: float = 0.40


@dataclass(frozen=True)
class TargetReuseItem:
    states: tuple[int, ...]
    actions: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class TargetReuseSummary:
    item_count: int
    avg_item_score: float
    best_item_score: float
    probe_success_rate: float


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


def evaluate_transfer_with_target_reuse(
    artifact: PretrainArtifact,
    target_spec: MiniGridSpec,
    seed: int,
    config: AdaptationConfig,
    reuse_config: TargetReuseConfig,
    scratch_final_score: float = 0.0,
) -> tuple[TransferReport, list[AdaptationPoint], TargetReuseSummary]:
    agent = artifact.best_agent()
    reuse_items, reuse_summary = build_target_reuse_items(target_spec, agent, seed + 7000, reuse_config)
    points = adapt_agent(target_spec, agent, seed, config, reuse_items=reuse_items, reuse_config=reuse_config)
    zero_shot = points[0].score if points else 0.0
    final_score = points[-1].score if points else 0.0
    lift = final_score - scratch_final_score
    return (
        TransferReport(
            method=f"{artifact.method}+target_reuse",
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
        reuse_summary,
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
    reuse_items: list[TargetReuseItem] | None = None,
    reuse_config: TargetReuseConfig | None = None,
) -> list[AdaptationPoint]:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec, episode_seed=seed)
    points = [_adaptation_point(spec, agent, seed + 10000, 0, config.eval_episodes)]
    reuse_horizon = int(config.episodes * (reuse_config.early_fraction if reuse_config is not None else 0.0))
    for episode in range(1, config.episodes + 1):
        if reuse_items and reuse_config is not None and episode <= reuse_horizon:
            _reinforce_target_reuse(agent, reuse_items, rng, reuse_config)
        env.episode_seed = seed + episode
        _run_episode(env, agent, rng, epsilon=config.epsilon, train=True)
        if episode % config.eval_every == 0 or episode == config.episodes:
            points.append(_adaptation_point(spec, agent, seed + 10000 + episode, episode, config.eval_episodes))
    env.close()
    return points


def evaluate_agent(spec: MiniGridSpec, agent: QAgent, seed: int, eval_episodes: int) -> AdaptationPoint:
    return _adaptation_point(spec, agent, seed, 0, eval_episodes)


def build_target_reuse_items(
    spec: MiniGridSpec,
    agent: QAgent,
    seed: int,
    config: TargetReuseConfig,
) -> tuple[list[TargetReuseItem], TargetReuseSummary]:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec)
    items = []
    successes = 0
    for idx in range(config.probe_episodes):
        env.episode_seed = seed + idx
        rollout = _run_episode(env, agent, rng, epsilon=config.probe_epsilon, train=False)
        successes += int(rollout.success)
        item = _target_reuse_item(rollout, spec.max_steps, config)
        if item is not None:
            items.append(item)
    env.close()
    scores = [item.score for item in items]
    return (
        items,
        TargetReuseSummary(
            item_count=len(items),
            avg_item_score=float(np.mean(scores)) if scores else 0.0,
            best_item_score=float(max(scores)) if scores else 0.0,
            probe_success_rate=successes / max(1, config.probe_episodes),
        ),
    )


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


def _target_reuse_item(
    rollout: Rollout,
    max_steps: int,
    config: TargetReuseConfig,
) -> TargetReuseItem | None:
    if len(rollout.actions) < 2:
        return None
    diagnostics = analyze_rollout(rollout, max_steps)
    if (
        not rollout.success
        and diagnostics.subgoal_score < config.min_subgoal_score
        and diagnostics.mobility < config.min_mobility
        and diagnostics.region_transition_count < 1
    ):
        return None
    success_bonus = 0.30 if rollout.success else 0.0
    score = (
        success_bonus
        + 0.36 * diagnostics.subgoal_score
        + 0.24 * min(1.0, diagnostics.region_transition_count / 6.0)
        + 0.20 * min(1.0, diagnostics.new_region_count / 4.0)
        + 0.20 * diagnostics.mobility
    )
    return TargetReuseItem(tuple(rollout.states), tuple(rollout.actions), score)


def _reinforce_target_reuse(
    agent: QAgent,
    items: list[TargetReuseItem],
    rng: np.random.Generator,
    config: TargetReuseConfig,
) -> None:
    if not items:
        return
    weights = np.array([max(0.01, item.score) for item in items], dtype=np.float64)
    weights = weights / weights.sum()
    for _ in range(config.reinforce_passes):
        item = items[int(rng.choice(len(items), p=weights))]
        reward = config.reward * (0.50 + item.score)
        for idx, action in enumerate(item.actions):
            state = item.states[idx]
            next_state = item.states[idx + 1]
            done = idx == len(item.actions) - 1
            agent.update(state, action, reward + (0.25 if done else 0.0), next_state, done)


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
