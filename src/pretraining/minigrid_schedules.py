from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from agents import QAgent, Rollout
from ecology import MotifBank, NicheArchive, ScoredMotifBank
from evolution import evolve_population
from minigrid_adapter import MiniGridSpec, MiniGridTabularEnv
from minigrid_diagnostics import analyze_rollout
from novelty import NoveltyArchive
from pretraining.artifacts import PretrainArtifact
from replay import SuccessReplayBank
from transfer.evaluator import evaluate_agent


PretrainBuilder = Callable[[MiniGridSpec, int, int, float], PretrainArtifact]


@dataclass(frozen=True)
class SourceEval:
    score: float
    success_rate: float
    avg_steps: float
    best_rollout: Rollout


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
        "cyclic_motif_fast_replay_pretrain": pretrain_cyclic_motif_fast_replay_agent,
        "cyclic_subgoal_ecology_replay_pretrain": pretrain_cyclic_subgoal_ecology_replay_agent,
    }
    try:
        builder = builders[method]
    except KeyError as exc:
        available = ", ".join(sorted(builders))
        msg = f"Unknown pretraining method {method!r}. Available: {available}"
        raise ValueError(msg) from exc
    return builder(spec, seed, episodes, epsilon)


def pretrain_cyclic_motif_fast_replay_agent(
    spec: MiniGridSpec,
    seed: int,
    episodes: int,
    epsilon: float,
) -> PretrainArtifact:
    rng = np.random.default_rng(seed)
    probe = MiniGridTabularEnv(spec, episode_seed=seed)
    population_size = _population_size(episodes)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(population_size)]
    probe.close()

    archive = NoveltyArchive(size=spec.max_steps)
    replay_bank = SuccessReplayBank(max_items=160)
    motif_bank = MotifBank(max_items=160)
    niche_archive = NicheArchive(max_per_cell=2)
    hall_of_fame = [agent.clone() for agent in population[: max(2, population_size // 4)]]
    schedule = ("anchored_explore", "operate", "free_explore", "operate", "fast_replay")
    fast_successes = 0
    total_successes = 0
    rollouts_used = 0
    source_score = 0.0

    env = MiniGridTabularEnv(spec)
    for generation in range(_generation_count(episodes, population_size)):
        phase = schedule[generation % len(schedule)]
        scores: list[float] = []
        for idx, agent in enumerate(population):
            if phase == "fast_replay":
                replay_bank.reinforce(agent, rng, passes=5, reward=0.11)
                motif_bank.reinforce(agent, rng, passes=5, reward=0.09)
                result = _evaluate_source_agent(spec, agent, seed + generation * 1000 + idx, eval_episodes=3)
                scores.append(_source_eval_score(result, spec.max_steps) + 0.05 * min(1.0, len(motif_bank) / 32.0))
                continue

            anchored = phase == "anchored_explore"
            if anchored and len(motif_bank):
                motif_bank.reinforce(agent, rng, passes=3, reward=0.08)
            phase_epsilon = min(epsilon, 0.14) if phase == "operate" else 0.28 if anchored else max(epsilon, 0.44)
            novelty_weight = 0.0 if phase == "operate" else 0.018 if anchored else 0.042
            rollout = run_minigrid_episode(
                env,
                agent,
                rng,
                epsilon=phase_epsilon,
                train=True,
                novelty_archive=archive,
                novelty_weight=novelty_weight,
                episode_seed=seed + generation * 1000 + idx,
            )
            archive.add(rollout.positions, rollout.success)
            total_successes += int(rollout.success)
            rollouts_used += 1
            if _is_fast_success(rollout, spec.max_steps):
                replay_bank.add(rollout)
                motif_bank.add_success(rollout)
                fast_successes += 1
            score = _rollout_score(rollout, spec.max_steps, archive.trajectory_novelty(rollout.positions))
            niche_archive.add(agent, rollout, score, spec.max_steps)
            scores.append(score + 0.06 * min(1.0, len(motif_bank) / 32.0))

        source_result = max(
            (_evaluate_source_agent(spec, agent, seed + 8000 + generation * 100 + idx, eval_episodes=3) for idx, agent in enumerate(population)),
            key=lambda result: result.score,
        )
        source_score = source_result.score
        hall_of_fame = _update_hall_of_fame(spec, hall_of_fame, population, seed + 12000 + generation * 100)
        population = evolve_population(
            population,
            scores,
            rng,
            elite_frac=0.35,
            mutation_scale=0.018 if phase in {"operate", "fast_replay"} else 0.070,
            mutation_rate=0.030 if phase in {"operate", "fast_replay"} else 0.080,
        )
        population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

    env.close()
    best = _best_agent(spec, population + hall_of_fame, seed + 20000)
    result = evaluate_agent(spec, best, seed + 9000, eval_episodes=6)
    return PretrainArtifact(
        method="cyclic_motif_fast_replay_pretrain",
        source_env=spec.env_id,
        seed=seed,
        population=[agent.clone() for agent in population],
        hall_of_fame=[agent.clone() for agent in hall_of_fame] + [best.clone()],
        metadata={
            "pretrain_episodes": episodes,
            "population_size": population_size,
            "rollouts_used": rollouts_used,
            "schedule": schedule,
            "fast_successes": fast_successes,
            "total_successes": total_successes,
            "fast_replay_ratio": fast_successes / max(1, total_successes),
            "replay_bank_size": len(replay_bank),
            "motif_count": len(motif_bank),
            "active_niches": len(niche_archive),
            "last_generation_source_score": source_score,
            "source_score": result.score,
            "source_success": result.success_rate,
        },
    )


def pretrain_cyclic_subgoal_ecology_replay_agent(
    spec: MiniGridSpec,
    seed: int,
    episodes: int,
    epsilon: float,
) -> PretrainArtifact:
    rng = np.random.default_rng(seed)
    probe = MiniGridTabularEnv(spec, episode_seed=seed)
    population_size = _population_size(episodes)
    population = [QAgent(probe.n_states, probe.n_actions, rng) for _ in range(population_size)]
    probe.close()

    archive = NoveltyArchive(size=spec.max_steps)
    replay_bank = SuccessReplayBank(max_items=180)
    motif_bank = ScoredMotifBank(max_items=200)
    niche_archive = NicheArchive(max_per_cell=2)
    hall_of_fame = [agent.clone() for agent in population[: max(2, population_size // 4)]]
    stress_survivors = [agent.clone() for agent in hall_of_fame]
    schedule = ("open_explore", "niche_sort", "operate", "stress_gate", "bottleneck", "subgoal_replay")
    subgoal_motifs = 0
    transition_motifs = 0
    confirmed_motifs = 0
    rollouts_used = 0
    stress_pass_rate = 0.0

    env = MiniGridTabularEnv(spec)
    for generation in range(_generation_count(episodes, population_size)):
        phase = schedule[generation % len(schedule)]

        if phase == "bottleneck":
            survivors = []
            survivors.extend(stress_survivors[: max(1, population_size // 4)])
            survivors.extend(niche_archive.local_survivors(per_cell=1, limit=max(2, population_size // 3)))
            survivors.extend([agent.clone() for agent in hall_of_fame])
            while len(survivors) < population_size:
                parent = survivors[int(rng.integers(len(survivors)))] if survivors else QAgent(probe.n_states, probe.n_actions, rng)
                child = parent.clone()
                child.mutate(scale=0.075, rate=0.085)
                survivors.append(child)
            population = survivors[:population_size]
            continue

        scores: list[float] = []
        stress_survivors = []
        for idx, agent in enumerate(population):
            if phase == "subgoal_replay":
                replay_bank.reinforce(agent, rng, passes=4, reward=0.09)
                motif_bank.reinforce(agent, rng, passes=5, reward=0.08)
                result = _evaluate_source_agent(spec, agent, seed + generation * 1000 + idx, eval_episodes=3)
                scores.append(_source_eval_score(result, spec.max_steps) + 0.05 * min(1.0, len(motif_bank) / 48.0))
                continue

            if phase == "stress_gate":
                result = _evaluate_source_agent(spec, agent, seed + generation * 1000 + idx, eval_episodes=3, epsilon=0.03)
                diagnostic = analyze_rollout(result.best_rollout, spec.max_steps).subgoal_score
                if result.success_rate > 0.0 or diagnostic >= 0.18:
                    stress_survivors.append(agent.clone())
                scores.append(0.55 * _source_eval_score(result, spec.max_steps) + 0.45 * diagnostic)
                continue

            if phase == "operate" and len(motif_bank) and rng.random() < 0.45:
                motif_bank.reinforce(agent, rng, passes=2, reward=0.065)
            phase_epsilon = max(epsilon, 0.48) if phase == "open_explore" else min(epsilon, 0.16)
            novelty_weight = 0.046 if phase == "open_explore" else 0.010 if phase == "niche_sort" else 0.0
            rollout = run_minigrid_episode(
                env,
                agent,
                rng,
                epsilon=phase_epsilon,
                train=True,
                novelty_archive=archive,
                novelty_weight=novelty_weight,
                revisit_penalty=0.010 if phase == "operate" else 0.0,
                turn_penalty=0.002 if phase == "operate" else 0.0,
                episode_seed=seed + generation * 1000 + idx,
            )
            archive.add(rollout.positions, rollout.success)
            novelty = archive.trajectory_novelty(rollout.positions)
            kind = _add_subgoal_memory(replay_bank, motif_bank, rollout, spec.max_steps, novelty)
            confirmed_motifs += int(kind is not None and kind.startswith("confirmed"))
            subgoal_motifs += int(kind is not None and kind.startswith("subgoal"))
            transition_motifs += int(kind == "subgoal_transition")
            rollouts_used += 1
            score = _subgoal_rollout_score(rollout, spec.max_steps, novelty, len(motif_bank), phase)
            niche_archive.add(agent, rollout, score, spec.max_steps)
            scores.append(score)

        if phase == "stress_gate":
            stress_pass_rate = len(stress_survivors) / max(1, population_size)
            if not stress_survivors:
                stress_survivors = niche_archive.best(max(1, population_size // 4)) or [agent.clone() for agent in hall_of_fame[:1]]

        hall_of_fame = _update_hall_of_fame(spec, hall_of_fame, population, seed + 22000 + generation * 100)
        population = evolve_population(
            population,
            scores,
            rng,
            elite_frac=0.35,
            mutation_scale=0.024 if phase in {"operate", "subgoal_replay", "stress_gate"} else 0.090,
            mutation_rate=0.035 if phase in {"operate", "subgoal_replay", "stress_gate"} else 0.100,
        )
        population[: len(hall_of_fame)] = [agent.clone() for agent in hall_of_fame]

    env.close()
    best = _best_agent(spec, population + hall_of_fame, seed + 30000)
    result = evaluate_agent(spec, best, seed + 9000, eval_episodes=6)
    candidate = motif_bank.kind_count("candidate") + motif_bank.kind_count("subgoal")
    return PretrainArtifact(
        method="cyclic_subgoal_ecology_replay_pretrain",
        source_env=spec.env_id,
        seed=seed,
        population=[agent.clone() for agent in population],
        hall_of_fame=[agent.clone() for agent in hall_of_fame] + [best.clone()],
        metadata={
            "pretrain_episodes": episodes,
            "population_size": population_size,
            "rollouts_used": rollouts_used,
            "schedule": schedule,
            "replay_bank_size": len(replay_bank),
            "motif_count": len(motif_bank),
            "confirmed_motif_count": motif_bank.kind_count("confirmed"),
            "candidate_motif_count": candidate,
            "subgoal_motif_count": subgoal_motifs,
            "transition_motif_count": transition_motifs,
            "confirmed_motif_events": confirmed_motifs,
            "stress_pass_rate": stress_pass_rate,
            "active_niches": len(niche_archive),
            "source_score": result.score,
            "source_success": result.success_rate,
        },
    )


def run_minigrid_episode(
    env: MiniGridTabularEnv,
    agent: QAgent,
    rng: np.random.Generator,
    epsilon: float,
    train: bool,
    novelty_archive: NoveltyArchive | None = None,
    novelty_weight: float = 0.0,
    revisit_penalty: float = 0.0,
    turn_penalty: float = 0.0,
    episode_seed: int | None = None,
) -> Rollout:
    if episode_seed is not None:
        env.episode_seed = episode_seed
    state = env.reset()
    states = [state]
    positions = [env.position]
    actions: list[int] = []
    rewards: list[float] = []
    visited = {env.position}
    success = False

    while True:
        if rng.random() < epsilon:
            action = int(rng.integers(env.n_actions))
        else:
            action = agent.act(state, 0.0)
        next_state, reward, done, info = env.step(action)
        position = info["position"]
        if novelty_archive is not None and novelty_weight:
            reward += novelty_weight * novelty_archive.position_bonus(position)
        if revisit_penalty and position in visited:
            reward -= revisit_penalty
        if turn_penalty and actions and action != actions[-1]:
            reward -= turn_penalty
        if train:
            agent.update(state, action, reward, next_state, done)
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(position)
        visited.add(position)
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


def _population_size(episodes: int) -> int:
    if episodes < 32:
        return 6
    if episodes < 96:
        return 8
    return 10


def _generation_count(episodes: int, population_size: int) -> int:
    return max(1, episodes // max(1, population_size))


def _is_fast_success(rollout: Rollout, max_steps: int) -> bool:
    return rollout.success and rollout.steps <= max(16, int(max_steps * 0.12))


def _evaluate_source_agent(
    spec: MiniGridSpec,
    agent: QAgent,
    seed: int,
    eval_episodes: int,
    epsilon: float = 0.0,
) -> SourceEval:
    rng = np.random.default_rng(seed)
    env = MiniGridTabularEnv(spec)
    rollouts = []
    for idx in range(eval_episodes):
        rollouts.append(run_minigrid_episode(env, agent, rng, epsilon=epsilon, train=False, episode_seed=seed + idx))
    env.close()
    successes = [rollout for rollout in rollouts if rollout.success]
    success_rate = len(successes) / max(1, len(rollouts))
    avg_steps = float(np.mean([rollout.steps for rollout in successes])) if successes else float(spec.max_steps)
    best_rollout = min(rollouts, key=lambda rollout: (not rollout.success, rollout.steps, -len(set(rollout.positions))))
    return SourceEval(
        score=0.55 * success_rate + 0.30 * (1.0 - min(avg_steps, spec.max_steps) / spec.max_steps),
        success_rate=success_rate,
        avg_steps=avg_steps,
        best_rollout=best_rollout,
    )


def _source_eval_score(result: SourceEval, max_steps: int) -> float:
    speed = 1.0 - min(result.avg_steps, max_steps) / max_steps
    diagnostic = analyze_rollout(result.best_rollout, max_steps).subgoal_score
    return 0.55 * result.success_rate + 0.25 * speed + 0.20 * diagnostic


def _rollout_score(rollout: Rollout, max_steps: int, novelty: float) -> float:
    success = float(rollout.success)
    speed = 1.0 - min(rollout.steps, max_steps) / max_steps
    diagnostic = analyze_rollout(rollout, max_steps).subgoal_score
    return 0.36 * success + 0.28 * speed + 0.20 * diagnostic + 0.16 * novelty


def _subgoal_rollout_score(
    rollout: Rollout,
    max_steps: int,
    novelty: float,
    motif_count: int,
    phase: str,
) -> float:
    success = float(rollout.success)
    speed = 1.0 - min(rollout.steps, max_steps) / max_steps
    diagnostics = analyze_rollout(rollout, max_steps)
    motif_reuse = min(1.0, motif_count / 48.0)
    if phase == "operate":
        return 0.34 * success + 0.26 * speed + 0.22 * diagnostics.subgoal_score + 0.12 * motif_reuse + 0.06 * novelty
    return (
        0.30 * novelty
        + 0.26 * diagnostics.subgoal_score
        + 0.18 * min(1.0, diagnostics.new_region_count / 6.0)
        + 0.14 * motif_reuse
        + 0.12 * speed
    )


def _add_subgoal_memory(
    replay_bank: SuccessReplayBank,
    motif_bank: ScoredMotifBank,
    rollout: Rollout,
    max_steps: int,
    novelty: float,
) -> str | None:
    kind = motif_bank.add_rollout(rollout, max_steps, novelty)
    if kind is not None and kind.startswith("confirmed"):
        replay_bank.add(rollout)
        return kind
    diagnostics = analyze_rollout(rollout, max_steps)
    return motif_bank.add_subgoal(
        rollout,
        diagnostics.subgoal_score,
        diagnostics.region_transition_count,
        diagnostics.new_region_count,
        novelty,
    )


def _update_hall_of_fame(
    spec: MiniGridSpec,
    hall_of_fame: list[QAgent],
    population: list[QAgent],
    seed: int,
    limit: int = 4,
) -> list[QAgent]:
    candidates = hall_of_fame + [agent.clone() for agent in population]
    scored = [
        (_evaluate_source_agent(spec, agent, seed + idx * 17, eval_episodes=3).score, agent)
        for idx, agent in enumerate(candidates)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [agent.clone() for _score, agent in scored[:limit]]


def _best_agent(spec: MiniGridSpec, population: list[QAgent], seed: int) -> QAgent:
    scored = [
        (_evaluate_source_agent(spec, agent, seed + idx * 17, eval_episodes=4).score, agent)
        for idx, agent in enumerate(population)
    ]
    return max(scored, key=lambda item: item[0])[1].clone()
