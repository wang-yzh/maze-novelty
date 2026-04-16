# Task Suite Transition Plan

## Purpose

The FourRooms benchmark has served its initial purpose:

- it showed that cyclic novelty/replay ideas are viable,
- it identified `cyclic_operate_replay` as a strong engineering baseline,
- it identified `cyclic_motif_fast_replay` as the current maze champion,
- it exposed the failure mode of richer memory systems that admit noisy motifs.

The next phase should not keep optimizing FourRooms alone. The goal is to build
a small task ladder that diagnoses where each mechanism works.

## Frozen Methods

Use a small fixed method set:

```text
cyclic_operate_replay
cyclic_motif_fast_replay
cyclic_ecology
cyclic_motif_bootstrap_replay
```

Roles:

- `cyclic_operate_replay`: simple exploitation baseline.
- `cyclic_motif_fast_replay`: current maze champion.
- `cyclic_ecology`: complex-task exploration/survival candidate.
- `cyclic_motif_bootstrap_replay`: cold-start candidate for sparse-success tasks.

Do not add more methods to the first task-suite pass.

## Task Ladder

Start with interpretable MiniGrid tasks:

```text
MiniGrid-FourRooms-v0
MiniGrid-MultiRoom-N4-S5-v0
MiniGrid-Dynamic-Obstacles-8x8-v0
```

Fallback rule:

If a task id is unavailable locally, list available MiniGrid ids with
`scripts/list_minigrid_envs.py` and substitute the nearest available task.

DoorKey-style tasks are intentionally excluded from the first pass because the
current MiniGrid adapter uses navigation actions only. DoorKey requires
pickup/toggle actions and should be introduced later with an explicit action-set
upgrade.

## Fixed Protocol

Default task-suite protocol:

```text
state_encoder = geometry
seeds = 7, 17, 27
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 100
```

For first contact with a new task, run a single seed first:

```text
seeds = 7
```

Only expand to three seeds when the single-seed result is informative and stable.

## Diagnostic Questions

The task suite should answer:

- Does `cyclic_motif_fast_replay` remain strong when full fast successes become
  rarer?
- Does `cyclic_ecology` become more competitive when success is sparse or moving
  obstacles increase robustness pressure?
- Does `cyclic_motif_bootstrap_replay` avoid a zero-motif cold start without
  polluting replay like v0.9?
- Is failure caused by the method, the state encoder, or the task interface?

## Decision Rules

Promote a challenger only if it beats or matches the champion on at least two of:

```text
test_score
test_success
test_avg_steps
nonzero_success_count
motif_bank_fill_generation
```

Retire or pause a challenger if:

```text
replay_improvement_delta < 0 across most seeds
candidate/subgoal motifs fill the bank but do not improve score
success rate drops relative to cyclic_motif_fast_replay
```

## Current Hypothesis

`cyclic_motif_fast_replay` is an exploitation champion when fast successes are
available. `cyclic_ecology` and `cyclic_motif_bootstrap_replay` may become more
valuable when complete success is sparse and the task requires staged discovery.

## First Single-Seed Result

Setup:

```text
seed = 7
state_encoder = geometry
method_time_limit_seconds = 100
methods = cyclic_operate_replay,
          cyclic_motif_fast_replay,
          cyclic_ecology,
          cyclic_motif_bootstrap_replay
```

Results:

| Task | Best Method | Test Success | Avg Steps | Test Score | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `MiniGrid-FourRooms-v0` | `cyclic_motif_bootstrap_replay` | `0.3333` | `7.5000` | `0.4745` | Bootstrap did not need candidate motifs; fast successes were available. |
| `MiniGrid-MultiRoom-N4-S5-v0` | none | `0.0000` | `256.0000` | `0.0000` | All methods failed; likely state/task interface bottleneck. |
| `MiniGrid-Dynamic-Obstacles-8x8-v0` | `cyclic_motif_bootstrap_replay` | `1.0000` | `19.6667` | `0.8270` | All methods solved; bootstrap and motif-fast were competitive. |

Interpretation:

- FourRooms is still useful as a regression check but no longer enough as a main
  optimization target.
- MultiRoom is the first truly diagnostic next task: all methods failed and
  motif banks stayed empty.
- Dynamic-Obstacles is solvable under the current navigation-only action set and
  can test speed/robustness, but it may be too easy at `8x8`.
- `cyclic_motif_bootstrap_replay` is viable, but in the successful tasks it won
  without needing bootstrap candidate motifs. Its cold-start value remains
  unproven.

Next useful task-suite changes:

```text
1. Add better progress/subgoal diagnostics for MultiRoom.
2. Consider a larger or harder Dynamic-Obstacles task.
3. Add an explicit action-set upgrade before DoorKey.
```
