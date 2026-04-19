# Version History

This document records the project history by git tag. It is meant to answer:

```text
What did each version add, what did it teach us, and what should we remember?
```

The project has gone through four major phases:

1. GridWorld proof of concept.
2. MiniGrid benchmark competition.
3. hard-task diagnostics and structural memory.
4. pretraining-transfer reframing.

## Phase 1: GridWorld Proof Of Concept

### `v0.1-gridworld-baseline`

Initial cyclic novelty experiments in custom GridWorld.

Added:

- tabular Q-learning agents;
- novelty archive;
- genetic population training;
- the first staged novelty/exploitation loop.

Lesson:

The cyclic idea was viable enough in small GridWorld to justify controlled
benchmarks. At this stage, the project was still framed as "solve this maze."

## Phase 2: MiniGrid Migration And Early Competition

### `v0.2-minigrid-pilot`

First MiniGrid benchmark pilot.

Added:

- MiniGrid runner;
- first bridge from custom GridWorld to external environments.

Lesson:

The project could leave the toy 12x12 maze, but environment/state encoding
became a central problem.

### `v0.3-minigrid-strong-cycle`

Added stronger MiniGrid cycle benchmark.

Lesson:

Cyclic training remained competitive after moving into MiniGrid. This made the
project look less like a one-map artifact.

### `v0.4-explore-operate-replay`

Added the explore-operate-replay benchmark.

Core idea:

```text
explore -> operate -> replay
```

Lesson:

This became the cleanest early cycle baseline. It had fewer moving parts than
ecology-style designs and stayed strong in direct maze-solving benchmarks.

### `v0.4.1-runtime-profiling`

Added MiniGrid runtime profiling.

Lesson:

Runtime became a real constraint on the local machine. This later pushed the
project toward time-limited benchmarks and smaller diagnostic runs.

### `v0.4.2-time-limited-benchmark`

Added time-limited MiniGrid benchmark.

Lesson:

Fixed-generation comparisons were too misleading when methods had different
runtime costs. Time budgets became part of fair evaluation.

## Phase 3: Framework Core And State Encoding

### `v0.5-framework-core`

Started the framework upgrade.

Added:

- method abstractions;
- structured benchmark recording;
- cleaner experiment plumbing.

Lesson:

The codebase needed to become an experiment framework, not a pile of one-off
runners.

### `v0.5.1-state-encoders`

Added MiniGrid state encoders.

Lesson:

State representation is not a detail. It controls whether tabular policies can
transfer any useful structure at all.

### `v0.5.2-state-comparison`

Recorded state encoder comparison.

Lesson:

The `geometry` encoder became important because it preserved more reusable
spatial structure than very compact encodings.

## Phase 4: Ecological Cycles

### `v0.6-ecological-cycle`

Added the first ecological cyclic method.

Core idea:

```text
open novelty -> niche sorting -> stress -> bottleneck -> re-radiation
```

Lesson:

The ecology idea was conceptually rich, but also introduced a much larger
hyperparameter surface.

### `v0.6.1-ecology-benchmark`

Recorded ecological cycle benchmark.

Lesson:

Ecology was promising, but did not clearly dominate the simpler
operate-replay baseline.

### `v0.6.2-ecology-ablations`

Added ecology ablation variants.

Lesson:

The project started separating "which component sounds plausible" from "which
component actually contributes."

### `v0.6.3-ecology-ablation-result`

Recorded ecology ablation benchmark.

Lesson:

The full ecology design was not automatically better. Some components produced
diagnostic signal but did not translate to final maze score.

## Phase 5: Lighter Ecological Replay Variants

### `v0.7.0-ecological-replay-variants`

Added lighter ecology-inspired variants.

Representative methods:

- `cyclic_speciated_stress_replay`;
- `cyclic_motif_oriented_radiation`;
- `cyclic_resource_ecology_replay`.

Lesson:

Splitting the large ecology idea into smaller variants was the right move. It
made the search space more understandable.

### `v0.7.1-variant-benchmark`

Recorded variant benchmark.

Lesson:

`cyclic_motif_oriented_radiation` became a meaningful challenger, while
`cyclic_operate_replay` remained the clean speed baseline.

## Phase 6: Motif Replay

### `v0.8.0-speed-disciplined-motif-replay`

Added speed-disciplined motif replay.

Core idea:

```text
only fast successes should become high-value replay/motif memory
```

Lesson:

This was one of the strongest direct maze-solving branches. It expressed the
original "fast and correct" instinct clearly.

### `v0.8.1-motif-fast-100s-benchmark`

Recorded 100-second benchmark for motif-fast replay.

Lesson:

`cyclic_motif_fast_replay` won the tested seeds in that benchmark. This became
one of the project's two most important historical winners.

### `v0.9.0-motif-memory-quality`

Added motif fast replay v2.

Added:

- scored motifs;
- medium success motifs;
- candidate motifs from high-progress failures.

Lesson:

More memory is not automatically better memory.

### `v0.9.1-motif-memory-quality-benchmark`

Recorded motif memory quality benchmark.

Lesson:

The richer motif bank filled, but did not improve over the simpler motif-fast
branch. Some candidate/medium motifs were likely too noisy.

## Phase 7: Task Suite Transition

### `v1.0.0-task-suite-transition`

Added task suite transition tools.

Lesson:

Single-task maze success had become too easy to overfit. The project needed a
suite of tasks.

### `v1.0.1-first-task-suite-result`

Recorded first task suite result.

Lesson:

MultiRoom exposed a hard failure mode. Methods that looked strong on FourRooms
did not automatically generalize.

## Phase 8: Diagnostics For Harder Tasks

### `v1.1.0-minigrid-evaluation-diagnostics`

Added MiniGrid evaluation diagnostics.

Metrics included:

- subgoal score;
- region transitions;
- new regions;
- revisit ratio;
- mobility.

Lesson:

When success is sparse, final score alone is too silent. Diagnostics became
necessary.

### `v1.2.0-subgoal-memory-diagnostics`

Added subgoal ecology replay diagnostics.

Lesson:

`cyclic_subgoal_ecology_replay` could fill memory from failed high-progress
trajectories. It still did not solve MultiRoom, but it proved failure memory is
real.

### Untagged: `Document MultiRoom diagnostic results`

Recorded MultiRoom diagnostic result after `v1.2.0`.

Lesson:

Subgoal memory generated structure, but not goal-directed navigation.

### Untagged: `Add structural novelty theory plan`

Added theory plan for structural novelty and options.

Lesson:

The project was starting to outgrow "maze solving" and needed a theory of
reusable behavioral structure.

### `v1.3.0-navigation-motif-diagnostics`

Added navigation motif diagnostics.

Lesson:

Navigation motifs could be extracted and measured, but this risked pulling the
project too deeply into maze-specific engineering.

## Phase 9: Pretraining-Transfer Reframing

### `v1.4.0-pretraining-transfer-smoke`

Added the pretraining transfer framework smoke test.

New central question:

```text
Does this evolutionary cycle create inherited structure that improves future adaptation?
```

Lesson:

This was the key conceptual reset. Maze is now an evaluation substrate for
pretraining effects, not the final objective.

### `v1.4.1-operate-replay-transfer-adapter`

Added `operate_replay_pretrain`.

Lesson:

The old clean winner could now participate in transfer evaluation as a
pretraining artifact.

### `v1.4.2-motif-subgoal-transfer-adapters`

Added:

- `cyclic_motif_fast_replay_pretrain`;
- `cyclic_subgoal_ecology_replay_pretrain`.

Lesson:

The two most important historical branches became comparable under the same
pretraining-transfer protocol.

### `v1.4.3-transfer-diagnostic-v1`

Added multi-seed transfer diagnostic runner.

Result:

FourRooms -> MultiRoom remained zero-success under the tested budget.

Lesson:

The artifact layer was informative even when final target score was all zero.

### `v1.4.4-transfer-ladder-probes`

Added target ladder and target-side probes.

Added:

- `--target-envs`;
- target probe subgoal score;
- target probe region transitions;
- target probe mobility.

Lesson:

Different artifacts produced different target-side behavior before success
appeared. This was the first transfer-frame evidence that artifact type affects
the target environment.

### `v1.4.5-target-reuse-adaptation`

Added target reuse adaptation.

Added:

- target reuse item extraction;
- early target trace reinforcement;
- deterministic agent cloning fix.

Result:

Target reuse items existed and had measurable structure, but simple trace
reinforcement still did not produce target success.

Lesson:

The bottleneck moved deeper:

```text
not "can we find reusable target-side traces?"
but "can we compose or execute them as reusable skills/options?"
```

### `v1.4.6-version-history-and-tests`

Added project history and reproducibility tests.

Added:

- `docs/version_history.md`;
- regression tests for agent cloning and transfer metric helpers;
- README and versioning updates for the standard quality-check commands.

Lesson:

The project needed a consolidation checkpoint. Before adding more mechanisms,
the historical record and minimal reproducibility checks had to catch up with
the pace of experimentation.

### Untagged: `behavior prior transfer scaffold`

Added:

- `StateSignature` for MiniGrid transfer-time structure matching;
- `BehaviorPrior` and `BehaviorLibrary`;
- target-side navigation prior extraction;
- early adaptation prior execution in addition to trace reinforcement;
- execution diagnostics for matched and executed priors.

Result:

Quick probe runs on `N2-S4` and `N4-S5` remained zero-success, but the new
executor was no longer silent: matched-prior counts, executed-prior counts, and
executed-prior steps were all substantial for the stronger source artifacts.

Lesson:

The bottleneck moved one step deeper again:

```text
not "can target-side priors be executed?"
but "how should executed priors change adaptation dynamics enough to create success?"
```

### Untagged: `prior-kind lifecycle diagnostics`

Added:

- `PriorKindStats` for per-`BehaviorPrior.kind` lifecycle tracking;
- compact JSON fields for target reuse item kind counts and execution outcomes;
- tests proving per-kind stats stay aligned with the existing aggregate counts.

Result:

The zero-score wall remains, but the failure is now more legible:

```text
motif priors fail mainly through effect and structural guards;
ecology priors fail mainly through progress guard loss;
operate open-loop priors complete often but remain semantically weak.
```

Lesson:

The next breakthrough is unlikely to come from another global continuation rule.
The project needs semantic behavior priors: intent, effect contract, and local
re-execution policy rather than only state match plus action trace.

### Untagged: `semantic behavior prior prototype`

Added:

- `semantic_intents` target-reuse execution mode;
- semantic executors for `forward_run`, `region_transition`, and `unstuck`;
- local geometry action selection from the current state signature;
- trace-as-budget execution for supported semantic priors;
- semantic execution CLI override through `--reuse-execution-mode`;
- tests for semantic selection, semantic action choice, and library fallback.

Result:

Hard-task final scores remained zero, but the structural execution signal
improved sharply in the quick diagnostic:

```text
motif N4-S5: completed 26 -> 94, aborted 63 -> 0
motif N4-S5: avg transitions 9.00 -> 16.75
ecology N4-S5: avg transitions 9.25 -> 22.75
```

Lesson:

The action-trace representation was a real bottleneck. Semantic execution can
turn fragile priors into sustained local behavior, but the next unsolved problem
is composition: movement has to become solved episodes.

### Untagged: `semantic composition diagnostics`

Added:

- composition outcome tracking for completed, aborted, and truncated priors;
- aggregate CSV metrics for region changes, goal visibility gains, displacement,
  and signature progress delta;
- per-kind composition metrics inside `target_reuse_prior_kind_stats_json`.

Result:

Semantic priors clearly create local movement and region changes, but this quick
run produced no goal visibility gains:

```text
N4-S5 motif: 97 outcomes, 25 region changes, 0 goal gains
N4-S5 ecology: 88 outcomes, 35 region changes, 0 goal gains
```

Lesson:

The bottleneck has moved from execution to composition. Semantic priors can
move through structure, but they are not yet target-directed enough to create
solved episodes.

### Untagged: `pretraining benefit longitudinal`

Added:

- a dedicated longitudinal benefit validation script;
- report, summary, and per-eval-step point outputs;
- random artifact control;
- shuffled prior control;
- paired adaptation seeds across no-reuse, true-reuse, and shuffled-reuse
  conditions;
- a reusable evaluator helper for externally supplied target reuse items.

Result:

The hard-target smoke still showed no task-level pretraining benefit:

```text
final_score = 0
adaptation_auc = 0
adaptation_auc_lift = 0
```

The same-source calibration produced a stronger warning: random artifact plus
target reuse can create transient AUC while the tested pretrained methods do
not. This means target-side reuse benefits cannot automatically be attributed
to pretraining.

Lesson:

The validation standard is now stricter: a claimed pretraining benefit must beat
scratch, random artifact control, and shuffled prior control across multiple
seeds.

### Untagged: `multiseed pretraining benefit result`

Recorded:

- a three-seed longitudinal validation over FourRooms, MultiRoom N2-S4, and
  MultiRoom N4-S5;
- `operate_replay_pretrain`, `cyclic_motif_fast_replay_pretrain`, and
  `cyclic_subgoal_ecology_replay_pretrain`;
- scratch, pretrained-agent, target-reuse, shuffled-prior, and random-artifact
  control conditions.

Result:

The run did not validate robust pretraining benefit.

FourRooms produced nonzero outcomes, but scratch had the strongest mean
adaptation AUC:

```text
scratch AUC = 0.2047
best target_reuse AUC = 0.1715
```

Both hard MultiRoom targets were all-zero at the task level:

```text
final_score = 0
adaptation_auc = 0
adaptation_auc_lift = 0
```

The strongest warning came from controls. Shuffled-prior and random-artifact
conditions could produce substantial region changes, sometimes matching or
exceeding true target reuse.

Lesson:

Region changes, prior executions, and other structural proxy metrics are now
diagnostics only. They are not evidence of transfer benefit unless they predict
task-level adaptation and beat scratch, shuffled, and random controls.

## Current State

Current branch:

```text
experiment/behavior-prior-transfer
```

Current latest tag:

```text
v1.4.6-version-history-and-tests
```

Current strongest historical ideas:

| Idea | Strength | Weakness |
| --- | --- | --- |
| `cyclic_operate_replay` | clean, fast, robust early baseline | does not explain sparse hard-task transfer |
| `cyclic_motif_fast_replay` | strong direct maze benchmark, expresses speed discipline | fast motif bank can stay empty in hard domains |
| `cyclic_subgoal_ecology_replay` | reliably creates failure-derived memory | memory does not yet compose into success |
| pretraining transfer framework | aligns with long-term finance goal | target adaptation mechanism is still immature |

## Strategic Lessons So Far

1. Direct maze success was useful early but is no longer enough.
2. Sparse hard tasks require diagnostics below final success.
3. Memory quantity is not memory quality.
4. Target-side behavior probes reveal differences that final score hides.
5. Trace replay alone is too weak; future work should test options, behavior
   priors, and subgoal-conditioned exploration.
6. Behavior-prior execution can now be observed directly; future work should
   measure whether executed priors improve mobility, transitions, or adaptation
   speed rather than only whether they exist.

## Recommended Next Direction

Do not add another large ecological cycle yet.

First stabilize a smaller research loop:

```text
pretrain artifact
target probe
candidate option extraction
early adaptation with option/behavior prior
transfer metrics
```

The next serious algorithmic question is:

```text
How should executed target-side priors change adaptation dynamics enough to
create transferable success?
```
