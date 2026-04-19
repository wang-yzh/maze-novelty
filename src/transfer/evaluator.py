from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from agents import QAgent, Rollout
from core.behavior import BehaviorLibrary, BehaviorPrior
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from minigrid_diagnostics import analyze_rollout
from minigrid_encoders import StateSignature
from navigation_motifs import NavigationMotif, extract_navigation_motifs
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
    match_mode: str = "strict"
    execution_mode: str = "default"
    min_subgoal_score: float = 0.018
    min_mobility: float = 0.035
    min_prior_quality: float = 0.28
    min_execution_support: int = 1
    allow_trace_priors: bool = True
    max_motifs_per_rollout: int = 3
    max_library_items: int = 24
    reinforce_passes: int = 4
    reward: float = 0.075
    early_fraction: float = 0.40
    execute_probability: float = 0.30
    execute_max_actions: int = 6
    abort_on_mismatch: bool = True
    mismatch_tolerance: int = 0
    continuation_rule: str = "signature"
    stall_tolerance: int = 0


@dataclass(frozen=True)
class TargetReuseItem:
    prior: BehaviorPrior

    @property
    def states(self) -> tuple[int, ...]:
        return self.prior.state_trace

    @property
    def actions(self) -> tuple[int, ...]:
        return self.prior.action_trace

    @property
    def score(self) -> float:
        return self.prior.score


@dataclass(frozen=True)
class TargetReuseSummary:
    item_count: int
    avg_item_score: float
    best_item_score: float
    avg_prior_support: float
    best_prior_support: int
    probe_success_rate: float
    matched_prior_count: int = 0
    executed_prior_count: int = 0
    completed_prior_count: int = 0
    truncated_prior_count: int = 0
    executed_prior_steps: int = 0
    first_step_mismatch_count: int = 0
    first_step_stall_count: int = 0
    mismatched_prior_steps: int = 0
    stalled_prior_steps: int = 0
    mismatch_abort_count: int = 0
    stall_abort_count: int = 0
    progress_lost_abort_count: int = 0
    aborted_prior_count: int = 0
    aborted_prior_steps: int = 0
    executed_episode_count: int = 0
    idle_episode_count: int = 0
    executed_episode_avg_subgoal_score: float = 0.0
    executed_episode_avg_region_transitions: float = 0.0
    executed_episode_avg_mobility: float = 0.0
    idle_episode_avg_subgoal_score: float = 0.0
    idle_episode_avg_region_transitions: float = 0.0
    idle_episode_avg_mobility: float = 0.0


@dataclass
class PriorExecutionStats:
    matched_prior_count: int = 0
    executed_prior_count: int = 0
    completed_prior_count: int = 0
    truncated_prior_count: int = 0
    executed_prior_steps: int = 0
    first_step_mismatch_count: int = 0
    first_step_stall_count: int = 0
    mismatched_prior_steps: int = 0
    stalled_prior_steps: int = 0
    mismatch_abort_count: int = 0
    stall_abort_count: int = 0
    progress_lost_abort_count: int = 0
    aborted_prior_count: int = 0
    aborted_prior_steps: int = 0
    executed_episode_count: int = 0
    idle_episode_count: int = 0
    executed_episode_subgoal_total: float = 0.0
    executed_episode_region_transition_total: float = 0.0
    executed_episode_mobility_total: float = 0.0
    idle_episode_subgoal_total: float = 0.0
    idle_episode_region_transition_total: float = 0.0
    idle_episode_mobility_total: float = 0.0


@dataclass
class ActivePriorExecution:
    prior: BehaviorPrior
    remaining_actions: list[int]
    remaining_signatures: list[StateSignature]
    match_mode: str
    consecutive_mismatches: int = 0
    consecutive_stalls: int = 0
    executed_steps: int = 0


@dataclass(frozen=True)
class PriorStepAssessment:
    mismatch: bool = False
    stalled: bool = False
    abort_reason: str = ""


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
    points, _execution_stats = adapt_agent(target_spec, agent, seed, config)
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
    points, execution_stats = adapt_agent(
        target_spec,
        agent,
        seed,
        config,
        reuse_items=reuse_items,
        reuse_config=reuse_config,
    )
    zero_shot = points[0].score if points else 0.0
    final_score = points[-1].score if points else 0.0
    lift = final_score - scratch_final_score
    reuse_summary = replace(
        reuse_summary,
        matched_prior_count=execution_stats.matched_prior_count,
        executed_prior_count=execution_stats.executed_prior_count,
        completed_prior_count=execution_stats.completed_prior_count,
        truncated_prior_count=execution_stats.truncated_prior_count,
        executed_prior_steps=execution_stats.executed_prior_steps,
        first_step_mismatch_count=execution_stats.first_step_mismatch_count,
        first_step_stall_count=execution_stats.first_step_stall_count,
        mismatched_prior_steps=execution_stats.mismatched_prior_steps,
        stalled_prior_steps=execution_stats.stalled_prior_steps,
        mismatch_abort_count=execution_stats.mismatch_abort_count,
        stall_abort_count=execution_stats.stall_abort_count,
        progress_lost_abort_count=execution_stats.progress_lost_abort_count,
        aborted_prior_count=execution_stats.aborted_prior_count,
        aborted_prior_steps=execution_stats.aborted_prior_steps,
        executed_episode_count=execution_stats.executed_episode_count,
        idle_episode_count=execution_stats.idle_episode_count,
        executed_episode_avg_subgoal_score=_average(
            execution_stats.executed_episode_subgoal_total,
            execution_stats.executed_episode_count,
        ),
        executed_episode_avg_region_transitions=_average(
            execution_stats.executed_episode_region_transition_total,
            execution_stats.executed_episode_count,
        ),
        executed_episode_avg_mobility=_average(
            execution_stats.executed_episode_mobility_total,
            execution_stats.executed_episode_count,
        ),
        idle_episode_avg_subgoal_score=_average(
            execution_stats.idle_episode_subgoal_total,
            execution_stats.idle_episode_count,
        ),
        idle_episode_avg_region_transitions=_average(
            execution_stats.idle_episode_region_transition_total,
            execution_stats.idle_episode_count,
        ),
        idle_episode_avg_mobility=_average(
            execution_stats.idle_episode_mobility_total,
            execution_stats.idle_episode_count,
        ),
    )
    abort_mode = (
        f"abort_t{reuse_config.mismatch_tolerance}"
        if reuse_config.abort_on_mismatch
        else "open_loop"
    )
    execution_mode = (
        abort_mode
        if reuse_config.execution_mode == "default"
        else f"{abort_mode}|{reuse_config.execution_mode}"
    )
    if reuse_config.continuation_rule != "signature":
        execution_mode = f"{execution_mode}|continue_{reuse_config.continuation_rule}"
    return (
        TransferReport(
            method=f"{artifact.method}+target_reuse[{reuse_config.match_mode}|{execution_mode}]",
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
    points, _execution_stats = adapt_agent(target_spec, agent, seed, config)
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
) -> tuple[list[AdaptationPoint], PriorExecutionStats]:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec, episode_seed=seed)
    points = [_adaptation_point(spec, agent, seed + 10000, 0, config.eval_episodes)]
    execution_stats = PriorExecutionStats()
    reuse_horizon = int(config.episodes * (reuse_config.early_fraction if reuse_config is not None else 0.0))
    prior_library = _build_prior_library(reuse_items, reuse_config) if reuse_config is not None else None
    for episode in range(1, config.episodes + 1):
        use_priors = bool(reuse_items and reuse_config is not None and episode <= reuse_horizon)
        active_reuse_items = reuse_items if use_priors else None
        active_reuse_config = reuse_config if use_priors else None
        episode_execution_stats = PriorExecutionStats()
        if use_priors:
            assert active_reuse_items is not None
            assert active_reuse_config is not None
            _reinforce_target_reuse(agent, active_reuse_items, rng, active_reuse_config)
        env.episode_seed = seed + episode
        rollout = _run_episode(
            env,
            agent,
            rng,
            epsilon=config.epsilon,
            train=True,
            prior_library=prior_library if use_priors else None,
            execution_mode=active_reuse_config.execution_mode if active_reuse_config is not None else "default",
            prior_execute_prob=active_reuse_config.execute_probability if active_reuse_config is not None else 0.0,
            execute_max_actions=active_reuse_config.execute_max_actions if active_reuse_config is not None else 0,
            abort_on_mismatch=active_reuse_config.abort_on_mismatch if active_reuse_config is not None else False,
            mismatch_tolerance=active_reuse_config.mismatch_tolerance if active_reuse_config is not None else 0,
            continuation_rule=active_reuse_config.continuation_rule if active_reuse_config is not None else "signature",
            stall_tolerance=active_reuse_config.stall_tolerance if active_reuse_config is not None else 0,
            execution_stats=episode_execution_stats if use_priors else None,
        )
        if use_priors:
            _accumulate_execution_stats(execution_stats, episode_execution_stats)
            _accumulate_episode_diagnostics(execution_stats, rollout, spec.max_steps, episode_execution_stats)
        if episode % config.eval_every == 0 or episode == config.episodes:
            points.append(_adaptation_point(spec, agent, seed + 10000 + episode, episode, config.eval_episodes))
    env.close()
    return points, execution_stats


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
    library = BehaviorLibrary(max_items=config.max_library_items, match_mode=config.match_mode)
    successes = 0
    for idx in range(config.probe_episodes):
        env.episode_seed = seed + idx
        rollout = _run_episode(env, agent, rng, epsilon=config.probe_epsilon, train=False)
        successes += int(rollout.success)
        for prior in _target_reuse_priors(rollout, spec.max_steps, config):
            library.add(prior)
    env.close()
    items = [TargetReuseItem(prior) for prior in library.priors]
    scores = [item.score for item in items]
    supports = [item.prior.support for item in items]
    return (
        items,
        TargetReuseSummary(
            item_count=len(items),
            avg_item_score=float(np.mean(scores)) if scores else 0.0,
            best_item_score=float(max(scores)) if scores else 0.0,
            avg_prior_support=float(np.mean(supports)) if supports else 0.0,
            best_prior_support=int(max(supports)) if supports else 0,
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


def _target_reuse_priors(
    rollout: Rollout,
    max_steps: int,
    config: TargetReuseConfig,
) -> list[BehaviorPrior]:
    if len(rollout.actions) < 2:
        return []
    diagnostics = analyze_rollout(rollout, max_steps)
    if (
        not rollout.success
        and diagnostics.subgoal_score < config.min_subgoal_score
        and diagnostics.mobility < config.min_mobility
        and diagnostics.region_transition_count < 1
    ):
        return []
    motifs = extract_navigation_motifs(rollout)
    priors = []
    for motif in motifs[: config.max_motifs_per_rollout]:
        prior = _behavior_prior_from_motif(rollout, motif, diagnostics.subgoal_score)
        if prior is not None and prior.score >= config.min_prior_quality:
            priors.append(prior)
    if priors:
        return priors
    fallback = _fallback_trace_prior(rollout, diagnostics.subgoal_score, config)
    return [fallback] if fallback is not None else []


def _behavior_prior_from_motif(
    rollout: Rollout,
    motif: NavigationMotif,
    subgoal_score: float,
) -> BehaviorPrior | None:
    if not rollout.state_signatures:
        return None
    start = motif.start
    end = motif.end
    if start < 0 or end <= start or end >= len(rollout.state_signatures):
        return None
    action_trace = tuple(int(action) for action in motif.actions)
    if len(action_trace) < 2:
        return None
    state_trace = tuple(int(state) for state in rollout.states[start : end + 1])
    if len(state_trace) != len(action_trace) + 1:
        return None
    kind_bonus = {
        "region_transition": 0.14,
        "forward_run": 0.08,
        "wall_follow_left": 0.06,
        "wall_follow_right": 0.06,
        "unstuck": 0.05,
    }.get(motif.kind, 0.0)
    score = (
        kind_bonus
        + 0.62 * motif.quality
        + 0.18 * subgoal_score
        + 0.10 * min(1.0, motif.region_transitions)
        + 0.10 * (1.0 - motif.collision_proxy_rate)
    )
    return BehaviorPrior(
        kind=motif.kind,
        initiation=rollout.state_signatures[start],
        action_trace=action_trace,
        termination=rollout.state_signatures[end],
        score=max(0.0, min(1.0, score)),
        source="target_probe_navigation",
        state_trace=state_trace,
        signature_trace=tuple(rollout.state_signatures[start : end + 1]),
    )


def _fallback_trace_prior(
    rollout: Rollout,
    subgoal_score: float,
    config: TargetReuseConfig,
) -> BehaviorPrior | None:
    if not rollout.state_signatures:
        return None
    action_count = min(len(rollout.actions), max(2, config.execute_max_actions))
    if action_count < 2 or action_count >= len(rollout.state_signatures):
        return None
    score = 0.22 * float(rollout.success) + 0.46 * subgoal_score + 0.16 * min(1.0, action_count / 6.0)
    if score < config.min_prior_quality:
        return None
    return BehaviorPrior(
        kind="target_trace",
        initiation=rollout.state_signatures[0],
        action_trace=tuple(int(action) for action in rollout.actions[:action_count]),
        termination=rollout.state_signatures[action_count],
        score=max(0.0, min(1.0, score)),
        source="target_probe_trace",
        state_trace=tuple(int(state) for state in rollout.states[: action_count + 1]),
        signature_trace=tuple(rollout.state_signatures[: action_count + 1]),
    )


def _build_prior_library(
    reuse_items: list[TargetReuseItem] | None,
    config: TargetReuseConfig,
) -> BehaviorLibrary | None:
    if not reuse_items or config.max_library_items <= 0:
        return None
    if config.execution_mode != "motif_fragments":
        library = BehaviorLibrary(max_items=config.max_library_items, match_mode=config.match_mode)
        for item in reuse_items:
            prior = item.prior
            if not config.allow_trace_priors and prior.kind == "target_trace":
                continue
            if prior.support < config.min_execution_support:
                continue
            library.add(prior)
        return library if len(library) else None

    # Motif-fragment mode should prefer repeated non-fallback fragments, but it
    # should not go silent if the budget is too small for high-support motifs.
    filter_stages = (
        (False, max(2, config.min_execution_support)),
        (False, 1),
        (True, 1),
    )
    for allow_trace_priors, min_support in filter_stages:
        library = BehaviorLibrary(max_items=config.max_library_items, match_mode=config.match_mode)
        for item in reuse_items:
            prior = item.prior
            if not allow_trace_priors and prior.kind == "target_trace":
                continue
            if prior.support < min_support:
                continue
            shortness_bonus = 0.04 * max(0.0, (6 - min(6, len(prior.action_trace))) / 6.0)
            support_bonus = 0.03 * min(3, max(0, prior.support - 1))
            library.add(replace(prior, score=min(1.0, prior.score + shortness_bonus + support_bonus)))
        if len(library):
            return library
    return None


def _accumulate_execution_stats(
    aggregate: PriorExecutionStats,
    episode: PriorExecutionStats,
) -> None:
    aggregate.matched_prior_count += episode.matched_prior_count
    aggregate.executed_prior_count += episode.executed_prior_count
    aggregate.completed_prior_count += episode.completed_prior_count
    aggregate.truncated_prior_count += episode.truncated_prior_count
    aggregate.executed_prior_steps += episode.executed_prior_steps
    aggregate.first_step_mismatch_count += episode.first_step_mismatch_count
    aggregate.first_step_stall_count += episode.first_step_stall_count
    aggregate.mismatched_prior_steps += episode.mismatched_prior_steps
    aggregate.stalled_prior_steps += episode.stalled_prior_steps
    aggregate.mismatch_abort_count += episode.mismatch_abort_count
    aggregate.stall_abort_count += episode.stall_abort_count
    aggregate.progress_lost_abort_count += episode.progress_lost_abort_count
    aggregate.aborted_prior_count += episode.aborted_prior_count
    aggregate.aborted_prior_steps += episode.aborted_prior_steps


def _accumulate_episode_diagnostics(
    aggregate: PriorExecutionStats,
    rollout: Rollout,
    max_steps: int,
    episode: PriorExecutionStats,
) -> None:
    diagnostics = analyze_rollout(rollout, max_steps)
    if episode.executed_prior_count > 0:
        aggregate.executed_episode_count += 1
        aggregate.executed_episode_subgoal_total += diagnostics.subgoal_score
        aggregate.executed_episode_region_transition_total += diagnostics.region_transition_count
        aggregate.executed_episode_mobility_total += diagnostics.mobility
        return

    aggregate.idle_episode_count += 1
    aggregate.idle_episode_subgoal_total += diagnostics.subgoal_score
    aggregate.idle_episode_region_transition_total += diagnostics.region_transition_count
    aggregate.idle_episode_mobility_total += diagnostics.mobility


def _average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


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
        if len(item.states) != len(item.actions) + 1:
            continue
        for idx, action in enumerate(item.actions):
            state = item.states[idx]
            next_state = item.states[idx + 1]
            done = idx == len(item.actions) - 1
            agent.update(state, action, reward + (0.25 if done else 0.0), next_state, done)


def _make_active_prior_execution(
    prior: BehaviorPrior,
    execute_max_actions: int,
    match_mode: str,
) -> ActivePriorExecution | None:
    limited_trace = list(int(action) for action in prior.action_trace[: max(1, execute_max_actions)])
    if not limited_trace:
        return None
    expected_signatures = []
    if len(prior.signature_trace) >= len(limited_trace) + 1:
        expected_signatures = list(prior.signature_trace[1 : len(limited_trace) + 1])
    return ActivePriorExecution(
        prior=prior,
        remaining_actions=limited_trace,
        remaining_signatures=expected_signatures,
        match_mode=match_mode,
    )


def _has_progress_evidence(signature: StateSignature) -> bool:
    return signature.goal_bin != 4 or signature.topology in (1, 2, 3)


def _assess_prior_execution_step(
    step_prior: ActivePriorExecution,
    action: int,
    expected_signature: StateSignature | None,
    observed_signature: StateSignature,
    previous_position: tuple[int, int],
    observed_position: tuple[int, int],
    abort_on_mismatch: bool,
    mismatch_tolerance: int,
    continuation_rule: str,
    stall_tolerance: int,
) -> PriorStepAssessment:
    stalled = action == 2 and observed_position == previous_position
    if stalled:
        step_prior.consecutive_stalls += 1
    else:
        step_prior.consecutive_stalls = 0

    mismatch = False
    if expected_signature is not None:
        mismatch = not expected_signature.matches(observed_signature, mode=step_prior.match_mode)
        if mismatch:
            step_prior.consecutive_mismatches += 1
        else:
            step_prior.consecutive_mismatches = 0

    if not abort_on_mismatch:
        return PriorStepAssessment(mismatch=mismatch, stalled=stalled)

    if continuation_rule == "signature":
        abort_reason = "mismatch" if mismatch and step_prior.consecutive_mismatches > mismatch_tolerance else ""
        return PriorStepAssessment(mismatch=mismatch, stalled=stalled, abort_reason=abort_reason)
    if continuation_rule == "motif_consistency":
        if stalled and step_prior.consecutive_stalls > stall_tolerance:
            return PriorStepAssessment(mismatch=mismatch, stalled=stalled, abort_reason="stall")
        abort_reason = "mismatch" if mismatch and step_prior.consecutive_mismatches > mismatch_tolerance else ""
        return PriorStepAssessment(mismatch=mismatch, stalled=stalled, abort_reason=abort_reason)
    if continuation_rule == "progress_guard":
        if stalled and step_prior.consecutive_stalls > stall_tolerance:
            return PriorStepAssessment(mismatch=mismatch, stalled=stalled, abort_reason="stall")
        abort_reason = "progress_lost" if (
            mismatch
            and step_prior.consecutive_mismatches > mismatch_tolerance
            and not _has_progress_evidence(observed_signature)
        ) else ""
        return PriorStepAssessment(mismatch=mismatch, stalled=stalled, abort_reason=abort_reason)

    msg = f"Unknown prior continuation rule: {continuation_rule}"
    raise ValueError(msg)


def _record_prior_step_assessment(
    stats: PriorExecutionStats,
    step_prior: ActivePriorExecution,
    assessment: PriorStepAssessment,
) -> None:
    is_first_step = step_prior.executed_steps == 0
    stats.executed_prior_steps += 1
    if assessment.mismatch:
        stats.mismatched_prior_steps += 1
        if is_first_step:
            stats.first_step_mismatch_count += 1
    if assessment.stalled:
        stats.stalled_prior_steps += 1
        if is_first_step:
            stats.first_step_stall_count += 1


def _record_prior_abort(stats: PriorExecutionStats, abort_reason: str) -> None:
    stats.aborted_prior_count += 1
    if abort_reason == "mismatch":
        stats.mismatch_abort_count += 1
    elif abort_reason == "stall":
        stats.stall_abort_count += 1
    elif abort_reason == "progress_lost":
        stats.progress_lost_abort_count += 1
    elif abort_reason:
        msg = f"Unknown prior abort reason: {abort_reason}"
        raise ValueError(msg)


def _select_prior_for_execution(
    prior_library: BehaviorLibrary,
    signature: StateSignature,
    execution_mode: str,
) -> BehaviorPrior | None:
    if execution_mode == "motif_fragments":
        candidates = [
            prior
            for prior in prior_library.priors
            if prior.matches(signature, mode=prior_library.match_mode)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda prior: (prior.support, -len(prior.action_trace), prior.score))
    return prior_library.best_match(signature)


def _run_episode(
    env: MiniGridTabularEnv,
    agent: QAgent,
    rng: np.random.Generator,
    epsilon: float,
    train: bool,
    prior_library: BehaviorLibrary | None = None,
    execution_mode: str = "default",
    prior_execute_prob: float = 0.0,
    execute_max_actions: int = 0,
    abort_on_mismatch: bool = False,
    mismatch_tolerance: int = 0,
    continuation_rule: str = "signature",
    stall_tolerance: int = 0,
    execution_stats: PriorExecutionStats | None = None,
) -> Rollout:
    state = env.reset()
    states = [state]
    positions = [env.position]
    state_signatures = [env.state_signature]
    actions = []
    rewards = []
    success = False
    active_prior: ActivePriorExecution | None = None
    while True:
        step_prior: ActivePriorExecution | None = None
        expected_signature = None
        executing_prior = active_prior is not None and bool(active_prior.remaining_actions)
        if executing_prior:
            assert active_prior is not None
            step_prior = active_prior
            action = active_prior.remaining_actions.pop(0)
            if active_prior.remaining_signatures:
                expected_signature = active_prior.remaining_signatures.pop(0)
        else:
            matched_prior = None
            if prior_library is not None and prior_execute_prob > 0.0:
                matched_prior = _select_prior_for_execution(prior_library, env.state_signature, execution_mode)
                if matched_prior is not None and execution_stats is not None:
                    execution_stats.matched_prior_count += 1
            if matched_prior is not None and rng.random() < prior_execute_prob and matched_prior.action_trace:
                active_prior = _make_active_prior_execution(
                    matched_prior,
                    execute_max_actions,
                    prior_library.match_mode if prior_library is not None else "strict",
                )
                if active_prior is not None:
                    step_prior = active_prior
                    action = active_prior.remaining_actions.pop(0)
                    if active_prior.remaining_signatures:
                        expected_signature = active_prior.remaining_signatures.pop(0)
                    if execution_stats is not None:
                        execution_stats.executed_prior_count += 1
                else:
                    action = agent.act(state, 0.0)
            elif rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:
                action = agent.act(state, 0.0)
        previous_position = env.position
        next_state, reward, done, info = env.step(action)
        if step_prior is not None:
            assessment = _assess_prior_execution_step(
                step_prior,
                action,
                expected_signature,
                info["state_signature"],
                previous_position,
                info["position"],
                abort_on_mismatch,
                mismatch_tolerance,
                continuation_rule,
                stall_tolerance,
            )
            if execution_stats is not None:
                _record_prior_step_assessment(execution_stats, step_prior, assessment)
            step_prior.executed_steps += 1
            if assessment.abort_reason:
                if execution_stats is not None:
                    _record_prior_abort(execution_stats, assessment.abort_reason)
                    execution_stats.aborted_prior_steps += len(step_prior.remaining_actions)
                active_prior = None
            elif step_prior is not None and not step_prior.remaining_actions:
                if execution_stats is not None:
                    execution_stats.completed_prior_count += 1
                active_prior = None
        if train:
            agent.update(state, action, reward, next_state, done)
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(info["position"])
        state_signatures.append(info["state_signature"])
        state = next_state
        success = bool(info["success"])
        if done:
            if active_prior is not None and active_prior.remaining_actions and execution_stats is not None:
                execution_stats.truncated_prior_count += 1
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
