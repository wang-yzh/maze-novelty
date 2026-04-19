from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

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
    effect_match_mode: str = "none"
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
    structural_patience: int = 2


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
    effect_described_item_count: int
    probe_success_rate: float
    item_kind_counts_json: str = "{}"
    prior_kind_stats_json: str = "{}"
    matched_prior_count: int = 0
    executed_prior_count: int = 0
    completed_prior_count: int = 0
    truncated_prior_count: int = 0
    executed_prior_steps: int = 0
    first_step_mismatch_count: int = 0
    first_step_stall_count: int = 0
    first_step_effect_mismatch_count: int = 0
    first_step_structural_evidence_count: int = 0
    mismatched_prior_steps: int = 0
    stalled_prior_steps: int = 0
    effect_mismatched_prior_steps: int = 0
    structural_evidence_steps: int = 0
    mismatch_abort_count: int = 0
    stall_abort_count: int = 0
    effect_abort_count: int = 0
    structural_abort_count: int = 0
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
class PriorKindStats:
    matched_prior_count: int = 0
    executed_prior_count: int = 0
    completed_prior_count: int = 0
    truncated_prior_count: int = 0
    executed_prior_steps: int = 0
    first_step_mismatch_count: int = 0
    first_step_stall_count: int = 0
    first_step_effect_mismatch_count: int = 0
    first_step_structural_evidence_count: int = 0
    mismatched_prior_steps: int = 0
    stalled_prior_steps: int = 0
    effect_mismatched_prior_steps: int = 0
    structural_evidence_steps: int = 0
    mismatch_abort_count: int = 0
    stall_abort_count: int = 0
    effect_abort_count: int = 0
    structural_abort_count: int = 0
    progress_lost_abort_count: int = 0
    aborted_prior_count: int = 0
    aborted_prior_steps: int = 0


@dataclass
class PriorExecutionStats:
    matched_prior_count: int = 0
    executed_prior_count: int = 0
    completed_prior_count: int = 0
    truncated_prior_count: int = 0
    executed_prior_steps: int = 0
    first_step_mismatch_count: int = 0
    first_step_stall_count: int = 0
    first_step_effect_mismatch_count: int = 0
    first_step_structural_evidence_count: int = 0
    mismatched_prior_steps: int = 0
    stalled_prior_steps: int = 0
    effect_mismatched_prior_steps: int = 0
    structural_evidence_steps: int = 0
    mismatch_abort_count: int = 0
    stall_abort_count: int = 0
    effect_abort_count: int = 0
    structural_abort_count: int = 0
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
    kind_stats: dict[str, PriorKindStats] = field(default_factory=dict)


@dataclass
class ActivePriorExecution:
    prior: BehaviorPrior
    remaining_actions: list[int]
    remaining_signatures: list[StateSignature]
    remaining_effects: list[str]
    match_mode: str
    start_position: tuple[int, int]
    consecutive_mismatches: int = 0
    consecutive_stalls: int = 0
    consecutive_effect_mismatches: int = 0
    consecutive_no_structural_evidence: int = 0
    executed_steps: int = 0


@dataclass(frozen=True)
class PriorStepAssessment:
    mismatch: bool = False
    stalled: bool = False
    effect_mismatch: bool = False
    structural_evidence: bool = False
    abort_reason: str = ""


SEMANTIC_INTENT_EXECUTION = "semantic_intents"
SEMANTIC_PRIOR_KINDS = frozenset({"forward_run", "region_transition", "unstuck"})


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
        first_step_effect_mismatch_count=execution_stats.first_step_effect_mismatch_count,
        first_step_structural_evidence_count=execution_stats.first_step_structural_evidence_count,
        mismatched_prior_steps=execution_stats.mismatched_prior_steps,
        stalled_prior_steps=execution_stats.stalled_prior_steps,
        effect_mismatched_prior_steps=execution_stats.effect_mismatched_prior_steps,
        structural_evidence_steps=execution_stats.structural_evidence_steps,
        mismatch_abort_count=execution_stats.mismatch_abort_count,
        stall_abort_count=execution_stats.stall_abort_count,
        effect_abort_count=execution_stats.effect_abort_count,
        structural_abort_count=execution_stats.structural_abort_count,
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
        prior_kind_stats_json=_prior_kind_stats_json(execution_stats),
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
    if reuse_config.effect_match_mode != "none":
        execution_mode = f"{execution_mode}|effect_{reuse_config.effect_match_mode}"
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
            effect_match_mode=active_reuse_config.effect_match_mode if active_reuse_config is not None else "none",
            prior_execute_prob=active_reuse_config.execute_probability if active_reuse_config is not None else 0.0,
            execute_max_actions=active_reuse_config.execute_max_actions if active_reuse_config is not None else 0,
            abort_on_mismatch=active_reuse_config.abort_on_mismatch if active_reuse_config is not None else False,
            mismatch_tolerance=active_reuse_config.mismatch_tolerance if active_reuse_config is not None else 0,
            continuation_rule=active_reuse_config.continuation_rule if active_reuse_config is not None else "signature",
            stall_tolerance=active_reuse_config.stall_tolerance if active_reuse_config is not None else 0,
            structural_patience=active_reuse_config.structural_patience if active_reuse_config is not None else 2,
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
            effect_described_item_count=sum(1 for item in items if item.prior.effect_trace),
            probe_success_rate=successes / max(1, config.probe_episodes),
            item_kind_counts_json=_item_kind_counts_json(items),
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
        effect_trace=_movement_effect_trace(action_trace, tuple(rollout.positions[start : end + 1])),
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
        effect_trace=_movement_effect_trace(
            tuple(int(action) for action in rollout.actions[:action_count]),
            tuple(rollout.positions[: action_count + 1]),
        ),
    )


def _build_prior_library(
    reuse_items: list[TargetReuseItem] | None,
    config: TargetReuseConfig,
) -> BehaviorLibrary | None:
    if not reuse_items or config.max_library_items <= 0:
        return None
    if config.execution_mode not in {"motif_fragments", SEMANTIC_INTENT_EXECUTION}:
        library = BehaviorLibrary(max_items=config.max_library_items, match_mode=config.match_mode)
        for item in reuse_items:
            prior = item.prior
            if not config.allow_trace_priors and prior.kind == "target_trace":
                continue
            if prior.support < config.min_execution_support:
                continue
            library.add(prior)
        return library if len(library) else None

    # Fragment-like modes should prefer repeated non-fallback fragments, but
    # they should not go silent if the budget is too small for high-support
    # motifs.
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
    aggregate.first_step_effect_mismatch_count += episode.first_step_effect_mismatch_count
    aggregate.first_step_structural_evidence_count += episode.first_step_structural_evidence_count
    aggregate.mismatched_prior_steps += episode.mismatched_prior_steps
    aggregate.stalled_prior_steps += episode.stalled_prior_steps
    aggregate.effect_mismatched_prior_steps += episode.effect_mismatched_prior_steps
    aggregate.structural_evidence_steps += episode.structural_evidence_steps
    aggregate.mismatch_abort_count += episode.mismatch_abort_count
    aggregate.stall_abort_count += episode.stall_abort_count
    aggregate.effect_abort_count += episode.effect_abort_count
    aggregate.structural_abort_count += episode.structural_abort_count
    aggregate.progress_lost_abort_count += episode.progress_lost_abort_count
    aggregate.aborted_prior_count += episode.aborted_prior_count
    aggregate.aborted_prior_steps += episode.aborted_prior_steps
    for kind, source in episode.kind_stats.items():
        target = _kind_stats(aggregate, kind)
        _accumulate_prior_kind_stats(target, source)


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


def _kind_stats(stats: PriorExecutionStats, kind: str) -> PriorKindStats:
    return stats.kind_stats.setdefault(kind, PriorKindStats())


def _accumulate_prior_kind_stats(target: PriorKindStats, source: PriorKindStats) -> None:
    target.matched_prior_count += source.matched_prior_count
    target.executed_prior_count += source.executed_prior_count
    target.completed_prior_count += source.completed_prior_count
    target.truncated_prior_count += source.truncated_prior_count
    target.executed_prior_steps += source.executed_prior_steps
    target.first_step_mismatch_count += source.first_step_mismatch_count
    target.first_step_stall_count += source.first_step_stall_count
    target.first_step_effect_mismatch_count += source.first_step_effect_mismatch_count
    target.first_step_structural_evidence_count += source.first_step_structural_evidence_count
    target.mismatched_prior_steps += source.mismatched_prior_steps
    target.stalled_prior_steps += source.stalled_prior_steps
    target.effect_mismatched_prior_steps += source.effect_mismatched_prior_steps
    target.structural_evidence_steps += source.structural_evidence_steps
    target.mismatch_abort_count += source.mismatch_abort_count
    target.stall_abort_count += source.stall_abort_count
    target.effect_abort_count += source.effect_abort_count
    target.structural_abort_count += source.structural_abort_count
    target.progress_lost_abort_count += source.progress_lost_abort_count
    target.aborted_prior_count += source.aborted_prior_count
    target.aborted_prior_steps += source.aborted_prior_steps


def _item_kind_counts_json(items: list[TargetReuseItem]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.prior.kind] = counts.get(item.prior.kind, 0) + 1
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _prior_kind_stats_json(stats: PriorExecutionStats) -> str:
    payload = {
        kind: _prior_kind_stats_dict(kind_stats)
        for kind, kind_stats in sorted(stats.kind_stats.items())
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _prior_kind_stats_dict(stats: PriorKindStats) -> dict[str, int]:
    return {
        "matched": stats.matched_prior_count,
        "started": stats.executed_prior_count,
        "completed": stats.completed_prior_count,
        "truncated": stats.truncated_prior_count,
        "aborted": stats.aborted_prior_count,
        "executed_steps": stats.executed_prior_steps,
        "aborted_steps": stats.aborted_prior_steps,
        "mismatch_steps": stats.mismatched_prior_steps,
        "stall_steps": stats.stalled_prior_steps,
        "effect_mismatch_steps": stats.effect_mismatched_prior_steps,
        "structural_evidence_steps": stats.structural_evidence_steps,
        "first_step_mismatches": stats.first_step_mismatch_count,
        "first_step_stalls": stats.first_step_stall_count,
        "first_step_effect_mismatches": stats.first_step_effect_mismatch_count,
        "first_step_structural_evidence": stats.first_step_structural_evidence_count,
        "mismatch_aborts": stats.mismatch_abort_count,
        "stall_aborts": stats.stall_abort_count,
        "effect_aborts": stats.effect_abort_count,
        "structural_aborts": stats.structural_abort_count,
        "progress_lost_aborts": stats.progress_lost_abort_count,
    }


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


def _movement_effect_trace(
    actions: tuple[int, ...],
    positions: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    effects = []
    for idx, action in enumerate(actions):
        previous = positions[idx] if idx < len(positions) else None
        observed = positions[idx + 1] if idx + 1 < len(positions) else previous
        effects.append(_movement_effect(action, previous, observed))
    return tuple(effects)


def _movement_effect(
    action: int,
    previous_position: tuple[int, int] | None,
    observed_position: tuple[int, int] | None,
) -> str:
    if action == 0:
        return "turn_left"
    if action == 1:
        return "turn_right"
    if action == 2:
        return "forward_move" if observed_position != previous_position else "forward_blocked"
    return "unknown"


def _make_active_prior_execution(
    prior: BehaviorPrior,
    execute_max_actions: int,
    match_mode: str,
    start_position: tuple[int, int],
) -> ActivePriorExecution | None:
    limited_trace = list(int(action) for action in prior.action_trace[: max(1, execute_max_actions)])
    if not limited_trace:
        return None
    expected_signatures = []
    if len(prior.signature_trace) >= len(limited_trace) + 1:
        expected_signatures = list(prior.signature_trace[1 : len(limited_trace) + 1])
    expected_effects = []
    if len(prior.effect_trace) >= len(limited_trace):
        expected_effects = list(prior.effect_trace[: len(limited_trace)])
    return ActivePriorExecution(
        prior=prior,
        remaining_actions=limited_trace,
        remaining_signatures=expected_signatures,
        remaining_effects=expected_effects,
        match_mode=match_mode,
        start_position=start_position,
    )


def _has_progress_evidence(signature: StateSignature) -> bool:
    return signature.goal_bin != 4 or signature.topology in (1, 2, 3)


def _position_region(position: tuple[int, int], region_size: int = 4) -> tuple[int, int]:
    row, col = position
    return row // region_size, col // region_size


def _manhattan_distance(position: tuple[int, int], other: tuple[int, int]) -> int:
    return abs(position[0] - other[0]) + abs(position[1] - other[1])


def _has_structural_evidence(
    step_prior: ActivePriorExecution,
    previous_position: tuple[int, int],
    observed_position: tuple[int, int],
    observed_signature: StateSignature,
) -> bool:
    if observed_signature.goal_bin != 4:
        return True
    if observed_position != previous_position:
        return True
    if _position_region(previous_position) != _position_region(observed_position):
        return True
    return _manhattan_distance(step_prior.start_position, observed_position) >= 2


def _has_kind_structural_evidence(
    step_prior: ActivePriorExecution,
    action: int,
    previous_position: tuple[int, int],
    observed_position: tuple[int, int],
    observed_signature: StateSignature,
) -> bool:
    kind = step_prior.prior.kind
    moved = observed_position != previous_position
    if observed_signature.goal_bin != 4:
        return True
    if kind == "forward_run":
        return action == 2 and moved
    if kind == "region_transition":
        return (
            _position_region(previous_position) != _position_region(observed_position)
            or _manhattan_distance(step_prior.start_position, observed_position) >= 2
        )
    if kind in {"wall_follow_left", "wall_follow_right"}:
        return action == 2 and moved
    if kind == "unstuck":
        return moved or _manhattan_distance(step_prior.start_position, observed_position) >= 1
    if kind == "target_trace":
        return False
    return _has_structural_evidence(step_prior, previous_position, observed_position, observed_signature)


def _is_passable_local_category(category: int) -> bool:
    return category in (0, 2)


def _uses_semantic_prior_execution(execution_mode: str, prior: BehaviorPrior) -> bool:
    return execution_mode == SEMANTIC_INTENT_EXECUTION and prior.kind in SEMANTIC_PRIOR_KINDS


def _semantic_prior_action(
    step_prior: ActivePriorExecution,
    signature: StateSignature,
) -> int:
    kind = step_prior.prior.kind
    if kind == "forward_run":
        return _semantic_forward_run_action(signature, step_prior.prior.action_trace)
    if kind == "region_transition":
        return _semantic_region_transition_action(signature, step_prior.prior.action_trace)
    if kind == "unstuck":
        return _semantic_unstuck_action(signature, step_prior.prior.action_trace)
    return int(step_prior.remaining_actions[0]) if step_prior.remaining_actions else 2


def _semantic_forward_run_action(
    signature: StateSignature,
    action_trace: tuple[int, ...],
) -> int:
    if _is_passable_local_category(signature.local_shape[0]):
        return 2
    return _turn_toward_open_side(signature, action_trace)


def _semantic_region_transition_action(
    signature: StateSignature,
    action_trace: tuple[int, ...],
) -> int:
    goal_action = _action_toward_visible_goal(signature)
    if goal_action is not None:
        return goal_action
    if _is_passable_local_category(signature.local_shape[0]):
        return 2
    return _turn_toward_open_side(signature, action_trace)


def _semantic_unstuck_action(
    signature: StateSignature,
    action_trace: tuple[int, ...],
) -> int:
    if _is_passable_local_category(signature.local_shape[0]):
        return 2
    return _turn_toward_open_side(signature, action_trace)


def _action_toward_visible_goal(signature: StateSignature) -> int | None:
    if signature.goal_bin == 4:
        return None
    horizontal = signature.goal_bin // 3
    vertical = signature.goal_bin % 3
    if horizontal < 1:
        return 0
    if horizontal > 1:
        return 1
    if vertical <= 1 and _is_passable_local_category(signature.local_shape[0]):
        return 2
    return None


def _turn_toward_open_side(
    signature: StateSignature,
    action_trace: tuple[int, ...],
) -> int:
    left_open = _is_passable_local_category(signature.local_shape[3])
    right_open = _is_passable_local_category(signature.local_shape[4])
    if left_open and not right_open:
        return 0
    if right_open and not left_open:
        return 1
    preferred_turn = _first_turn_action(action_trace)
    if preferred_turn is not None:
        return preferred_turn
    return 0 if left_open else 1


def _first_turn_action(action_trace: tuple[int, ...]) -> int | None:
    for action in action_trace:
        if action in (0, 1):
            return int(action)
    return None


def _expected_effect_from_signature(action: int, signature: StateSignature) -> str:
    if action == 0:
        return "turn_left"
    if action == 1:
        return "turn_right"
    if action == 2:
        return "forward_move" if _is_passable_local_category(signature.local_shape[0]) else "forward_blocked"
    return "unknown"


def _next_prior_execution_action(
    step_prior: ActivePriorExecution,
    execution_mode: str,
    signature: StateSignature,
) -> tuple[int, StateSignature | None, str | None]:
    action = step_prior.remaining_actions.pop(0)
    if _uses_semantic_prior_execution(execution_mode, step_prior.prior):
        semantic_action = _semantic_prior_action(step_prior, signature)
        return semantic_action, None, _expected_effect_from_signature(semantic_action, signature)

    expected_signature = None
    expected_effect = None
    if step_prior.remaining_signatures:
        expected_signature = step_prior.remaining_signatures.pop(0)
    if step_prior.remaining_effects:
        expected_effect = step_prior.remaining_effects.pop(0)
    return action, expected_signature, expected_effect


def _first_step_effect_matches(
    prior: BehaviorPrior,
    signature: StateSignature,
    effect_match_mode: str,
) -> bool:
    if effect_match_mode == "none":
        return True
    if effect_match_mode != "first_step":
        msg = f"Unknown effect match mode: {effect_match_mode}"
        raise ValueError(msg)
    if not prior.action_trace or not prior.effect_trace:
        return True

    action = prior.action_trace[0]
    effect = prior.effect_trace[0]
    if action == 2:
        front_passable = _is_passable_local_category(signature.local_shape[0])
        if effect == "forward_move":
            return front_passable
        if effect == "forward_blocked":
            return not front_passable
        return True
    if action == 0:
        if effect != "turn_left":
            return False
        if len(prior.signature_trace) < 2:
            return True
        return prior.signature_trace[1].local_shape[0] == signature.local_shape[3]
    if action == 1:
        if effect != "turn_right":
            return False
        if len(prior.signature_trace) < 2:
            return True
        return prior.signature_trace[1].local_shape[0] == signature.local_shape[4]
    return True


def _prior_effect_matches_for_execution(
    prior: BehaviorPrior,
    signature: StateSignature,
    execution_mode: str,
    effect_match_mode: str,
) -> bool:
    if _uses_semantic_prior_execution(execution_mode, prior):
        return True
    return _first_step_effect_matches(prior, signature, effect_match_mode)


def _assess_prior_execution_step(
    step_prior: ActivePriorExecution,
    action: int,
    expected_effect: str | None,
    expected_signature: StateSignature | None,
    observed_signature: StateSignature,
    previous_position: tuple[int, int],
    observed_position: tuple[int, int],
    abort_on_mismatch: bool,
    mismatch_tolerance: int,
    continuation_rule: str,
    stall_tolerance: int,
    structural_patience: int,
) -> PriorStepAssessment:
    stalled = action == 2 and observed_position == previous_position
    if stalled:
        step_prior.consecutive_stalls += 1
    else:
        step_prior.consecutive_stalls = 0

    observed_effect = _movement_effect(action, previous_position, observed_position)
    effect_mismatch = expected_effect is not None and expected_effect != observed_effect
    if effect_mismatch:
        step_prior.consecutive_effect_mismatches += 1
    else:
        step_prior.consecutive_effect_mismatches = 0

    mismatch = False
    if expected_signature is not None:
        mismatch = not expected_signature.matches(observed_signature, mode=step_prior.match_mode)
        if mismatch:
            step_prior.consecutive_mismatches += 1
        else:
            step_prior.consecutive_mismatches = 0

    if continuation_rule == "kind_structural_guard":
        structural_evidence = _has_kind_structural_evidence(
            step_prior,
            action,
            previous_position,
            observed_position,
            observed_signature,
        )
    else:
        structural_evidence = _has_structural_evidence(
            step_prior,
            previous_position,
            observed_position,
            observed_signature,
        )
    if structural_evidence:
        step_prior.consecutive_no_structural_evidence = 0
    else:
        step_prior.consecutive_no_structural_evidence += 1

    if not abort_on_mismatch:
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
        )

    if continuation_rule == "signature":
        abort_reason = "mismatch" if mismatch and step_prior.consecutive_mismatches > mismatch_tolerance else ""
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
            abort_reason=abort_reason,
        )
    if continuation_rule == "motif_consistency":
        if stalled and step_prior.consecutive_stalls > stall_tolerance:
            return PriorStepAssessment(
                mismatch=mismatch,
                stalled=stalled,
                effect_mismatch=effect_mismatch,
                structural_evidence=structural_evidence,
                abort_reason="stall",
            )
        abort_reason = "mismatch" if mismatch and step_prior.consecutive_mismatches > mismatch_tolerance else ""
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
            abort_reason=abort_reason,
        )
    if continuation_rule == "effect_consistency":
        abort_reason = "effect" if effect_mismatch else ""
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
            abort_reason=abort_reason,
        )
    if continuation_rule == "effect_structural_guard":
        if effect_mismatch:
            abort_reason = "effect"
        elif step_prior.consecutive_no_structural_evidence > structural_patience:
            abort_reason = "structural"
        else:
            abort_reason = ""
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
            abort_reason=abort_reason,
        )
    if continuation_rule == "kind_structural_guard":
        if effect_mismatch:
            abort_reason = "effect"
        elif step_prior.consecutive_no_structural_evidence > structural_patience:
            abort_reason = "structural"
        else:
            abort_reason = ""
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
            abort_reason=abort_reason,
        )
    if continuation_rule == "progress_guard":
        if stalled and step_prior.consecutive_stalls > stall_tolerance:
            return PriorStepAssessment(
                mismatch=mismatch,
                stalled=stalled,
                effect_mismatch=effect_mismatch,
                structural_evidence=structural_evidence,
                abort_reason="stall",
            )
        abort_reason = "progress_lost" if (
            mismatch
            and step_prior.consecutive_mismatches > mismatch_tolerance
            and not _has_progress_evidence(observed_signature)
        ) else ""
        return PriorStepAssessment(
            mismatch=mismatch,
            stalled=stalled,
            effect_mismatch=effect_mismatch,
            structural_evidence=structural_evidence,
            abort_reason=abort_reason,
        )

    msg = f"Unknown prior continuation rule: {continuation_rule}"
    raise ValueError(msg)


def _record_prior_match(stats: PriorExecutionStats, prior: BehaviorPrior) -> None:
    stats.matched_prior_count += 1
    _kind_stats(stats, prior.kind).matched_prior_count += 1


def _record_prior_start(stats: PriorExecutionStats, step_prior: ActivePriorExecution) -> None:
    stats.executed_prior_count += 1
    _kind_stats(stats, step_prior.prior.kind).executed_prior_count += 1


def _record_prior_step_assessment(
    stats: PriorExecutionStats,
    step_prior: ActivePriorExecution,
    assessment: PriorStepAssessment,
) -> None:
    is_first_step = step_prior.executed_steps == 0
    kind_stats = _kind_stats(stats, step_prior.prior.kind)
    stats.executed_prior_steps += 1
    kind_stats.executed_prior_steps += 1
    if assessment.mismatch:
        stats.mismatched_prior_steps += 1
        kind_stats.mismatched_prior_steps += 1
        if is_first_step:
            stats.first_step_mismatch_count += 1
            kind_stats.first_step_mismatch_count += 1
    if assessment.stalled:
        stats.stalled_prior_steps += 1
        kind_stats.stalled_prior_steps += 1
        if is_first_step:
            stats.first_step_stall_count += 1
            kind_stats.first_step_stall_count += 1
    if assessment.effect_mismatch:
        stats.effect_mismatched_prior_steps += 1
        kind_stats.effect_mismatched_prior_steps += 1
        if is_first_step:
            stats.first_step_effect_mismatch_count += 1
            kind_stats.first_step_effect_mismatch_count += 1
    if assessment.structural_evidence:
        stats.structural_evidence_steps += 1
        kind_stats.structural_evidence_steps += 1
        if is_first_step:
            stats.first_step_structural_evidence_count += 1
            kind_stats.first_step_structural_evidence_count += 1


def _record_prior_completion(stats: PriorExecutionStats, step_prior: ActivePriorExecution) -> None:
    stats.completed_prior_count += 1
    _kind_stats(stats, step_prior.prior.kind).completed_prior_count += 1


def _record_prior_truncation(stats: PriorExecutionStats, step_prior: ActivePriorExecution) -> None:
    stats.truncated_prior_count += 1
    _kind_stats(stats, step_prior.prior.kind).truncated_prior_count += 1


def _record_prior_abort(
    stats: PriorExecutionStats,
    step_prior: ActivePriorExecution,
    abort_reason: str,
    aborted_steps: int,
) -> None:
    kind_stats = _kind_stats(stats, step_prior.prior.kind)
    stats.aborted_prior_count += 1
    stats.aborted_prior_steps += aborted_steps
    kind_stats.aborted_prior_count += 1
    kind_stats.aborted_prior_steps += aborted_steps
    if abort_reason == "mismatch":
        stats.mismatch_abort_count += 1
        kind_stats.mismatch_abort_count += 1
    elif abort_reason == "stall":
        stats.stall_abort_count += 1
        kind_stats.stall_abort_count += 1
    elif abort_reason == "effect":
        stats.effect_abort_count += 1
        kind_stats.effect_abort_count += 1
    elif abort_reason == "structural":
        stats.structural_abort_count += 1
        kind_stats.structural_abort_count += 1
    elif abort_reason == "progress_lost":
        stats.progress_lost_abort_count += 1
        kind_stats.progress_lost_abort_count += 1
    elif abort_reason:
        msg = f"Unknown prior abort reason: {abort_reason}"
        raise ValueError(msg)


def _select_prior_for_execution(
    prior_library: BehaviorLibrary,
    signature: StateSignature,
    execution_mode: str,
    effect_match_mode: str,
) -> BehaviorPrior | None:
    candidates = [
        prior
        for prior in prior_library.priors
        if prior.matches(signature, mode=prior_library.match_mode)
        and _prior_effect_matches_for_execution(prior, signature, execution_mode, effect_match_mode)
    ]
    if not candidates:
        return None
    if execution_mode == SEMANTIC_INTENT_EXECUTION:
        return max(
            candidates,
            key=lambda prior: (
                prior.kind in SEMANTIC_PRIOR_KINDS,
                prior.score,
                prior.support,
                -len(prior.action_trace),
            ),
        )
    if execution_mode == "motif_fragments":
        return max(candidates, key=lambda prior: (prior.support, -len(prior.action_trace), prior.score))
    return max(candidates, key=lambda prior: (prior.score, prior.support, -len(prior.action_trace)))


def _run_episode(
    env: MiniGridTabularEnv,
    agent: QAgent,
    rng: np.random.Generator,
    epsilon: float,
    train: bool,
    prior_library: BehaviorLibrary | None = None,
    execution_mode: str = "default",
    effect_match_mode: str = "none",
    prior_execute_prob: float = 0.0,
    execute_max_actions: int = 0,
    abort_on_mismatch: bool = False,
    mismatch_tolerance: int = 0,
    continuation_rule: str = "signature",
    stall_tolerance: int = 0,
    structural_patience: int = 2,
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
        expected_effect = None
        executing_prior = active_prior is not None and bool(active_prior.remaining_actions)
        if executing_prior:
            assert active_prior is not None
            step_prior = active_prior
            action, expected_signature, expected_effect = _next_prior_execution_action(
                active_prior,
                execution_mode,
                env.state_signature,
            )
        else:
            matched_prior = None
            if prior_library is not None and prior_execute_prob > 0.0:
                matched_prior = _select_prior_for_execution(
                    prior_library,
                    env.state_signature,
                    execution_mode,
                    effect_match_mode,
                )
                if matched_prior is not None and execution_stats is not None:
                    _record_prior_match(execution_stats, matched_prior)
            if matched_prior is not None and rng.random() < prior_execute_prob and matched_prior.action_trace:
                active_prior = _make_active_prior_execution(
                    matched_prior,
                    execute_max_actions,
                    prior_library.match_mode if prior_library is not None else "strict",
                    env.position,
                )
                if active_prior is not None:
                    step_prior = active_prior
                    action, expected_signature, expected_effect = _next_prior_execution_action(
                        active_prior,
                        execution_mode,
                        env.state_signature,
                    )
                    if execution_stats is not None:
                        _record_prior_start(execution_stats, active_prior)
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
                expected_effect,
                expected_signature,
                info["state_signature"],
                previous_position,
                info["position"],
                abort_on_mismatch,
                mismatch_tolerance,
                continuation_rule,
                stall_tolerance,
                structural_patience,
            )
            if execution_stats is not None:
                _record_prior_step_assessment(execution_stats, step_prior, assessment)
            step_prior.executed_steps += 1
            if assessment.abort_reason:
                if execution_stats is not None:
                    _record_prior_abort(
                        execution_stats,
                        step_prior,
                        assessment.abort_reason,
                        len(step_prior.remaining_actions),
                    )
                active_prior = None
            elif step_prior is not None and not step_prior.remaining_actions:
                if execution_stats is not None:
                    _record_prior_completion(execution_stats, step_prior)
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
                _record_prior_truncation(execution_stats, active_prior)
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
