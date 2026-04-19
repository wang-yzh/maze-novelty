# Pretraining Transfer Framework

## Why We Are Reframing

The project drifted toward "solve harder mazes." That is not the original goal.

The original goal is:

```text
find evolutionary cycle models that produce strong pretraining effects
```

The long-term target is financial strategy learning, especially rolling adaptation under changing regimes. Maze and MiniGrid tasks are useful only if they help us evaluate transfer, adaptation speed, robustness, and reusable behavioral structure.

The new central question is:

```text
Which pretraining cycle creates the best initial condition for future adaptation?
```

Not:

```text
Which method solves this one maze?
```

## Old Evaluation Frame

The old frame:

```text
train on task -> evaluate final score
```

This naturally pushed the project toward task-specific fixes:

```text
wall-following
doorway crossing
maze topology
navigation options
```

Those can be useful diagnostics, but they should not become the main objective.

## New Evaluation Frame

The new frame:

```text
pretrain on source suite
export artifact
transfer to target suite
adapt under fixed budget
evaluate learning curve and stability
```

This better matches the future financial setup:

```text
pretrain on historical/simulated regimes
transfer to unseen regime
adapt with limited budget
evaluate return, drawdown, stability, and execution discipline
```

## Core Objects

### PretrainArtifact

A pretraining run should export an artifact, not just a final score.

Minimum artifact:

```text
method
source_env
seed
population
hall_of_fame
metadata
```

Later artifact fields:

```text
motif_bank
subgoal_memory
niche_archive
navigation_motifs
stress_survivors
training_curve
```

The artifact represents inherited structure.

### TransferReport

A transfer run should report adaptation behavior.

Minimum report:

```text
method
source_env
target_env
seed
zero_shot_score
adaptation_auc
time_to_first_success
time_to_threshold
final_score
transfer_lift
negative_transfer
```

This separates "good during pretraining" from "useful for later learning."

## Metrics

### Zero-Shot Score

Evaluate the artifact on the target task before target training.

Meaning:

```text
Did pretraining directly create useful behavior?
```

### Adaptation AUC

Area under the target adaptation learning curve.

Meaning:

```text
Did pretraining help learning happen earlier?
```

This should often matter more than final score.

### Time To First Success

First target adaptation step where success appears.

Meaning:

```text
Did pretraining reduce cold-start time?
```

### Time To Threshold

First target adaptation step where score reaches a predefined threshold.

Meaning:

```text
Did pretraining reach usable behavior faster?
```

### Transfer Lift

Difference between pretrained adaptation and scratch adaptation under the same budget.

Meaning:

```text
Was the artifact actually useful?
```

### Negative Transfer Rate

How often pretraining is worse than scratch.

Meaning:

```text
Does the pretraining cycle produce harmful bias?
```

## How Existing Work Maps Into The New Frame

### `cyclic_operate_replay`

Role:

```text
early strong pretraining schedule baseline
```

It should be evaluated by transfer lift, not only by FourRooms score.

### `cyclic_motif_fast_replay`

Role:

```text
success-fragment pretraining schedule
```

Known weakness:

```text
fails when success is sparse and motif bank stays empty
```

### `cyclic_subgoal_ecology_replay`

Role:

```text
failure-memory pretraining schedule
```

Known result:

```text
fills memory on MultiRoom but does not directly solve it
```

New question:

```text
Does its memory improve adaptation on a different target?
```

### `navigation_motifs`

Role:

```text
artifact diagnostic tool
```

It should not become the main task-specific solution until transfer usefulness is proven.

## First Minimal Refactor

The first implementation should be deliberately small:

```text
src/pretraining/artifacts.py
src/transfer/metrics.py
src/transfer/evaluator.py
scripts/run_pretraining_transfer_smoke.py
```

The first smoke test should compare:

```text
scratch adaptation
simple Q pretraining on source env -> target adaptation
```

This does not yet evaluate all cycle models. Its purpose is to prove the evaluation protocol works.

## First Cycle Adapter

The first existing cycle schedule has now been connected to the pretraining frame:

```text
operate_replay_pretrain
```

It is intentionally conservative:

```text
source env only
single inherited Q artifact
success replay bank populated only by successful source rollouts
phase loop: explore -> operate -> replay
```

The goal is not to fully reproduce every historical maze experiment. The goal is to make old winners participate in the new question:

```text
Does this schedule create a better starting condition for target adaptation?
```

This adapter is now the reference path for adding later schedules:

```text
cyclic_motif_fast_replay
cyclic_subgoal_ecology_replay
ecology variants
```

The next two adapters were added:

```text
cyclic_motif_fast_replay_pretrain
cyclic_subgoal_ecology_replay_pretrain
```

They intentionally preserve different hypotheses:

| Method | Pretraining Hypothesis |
| --- | --- |
| `cyclic_motif_fast_replay_pretrain` | fast successful fragments are the best inherited structure |
| `cyclic_subgoal_ecology_replay_pretrain` | high-progress failures can become useful inherited structure when true success is sparse |

This distinction matters for hard domains. If success is common, fast replay may dominate. If success is rare, subgoal ecology should at least avoid an empty memory bank.

## First Smoke Test

Recommended source:

```text
MiniGrid-FourRooms-v0
```

Recommended target:

```text
MiniGrid-MultiRoom-N4-S5-v0
```

Reason:

```text
FourRooms is learnable enough to create an artifact.
MultiRoom is hard enough to reveal whether the artifact transfers.
```

Expected outcome:

The first smoke test may show little or no positive transfer. That is acceptable. The goal is to make transfer measurable.

Initial command:

```bash
uv run python scripts/run_pretraining_transfer_smoke.py \
  --source-env MiniGrid-FourRooms-v0 \
  --target-env MiniGrid-MultiRoom-N4-S5-v0 \
  --seed 7 \
  --state-encoder geometry \
  --pretrain-episodes 20 \
  --adapt-episodes 20 \
  --eval-every 10 \
  --eval-episodes 3 \
  --output outputs/pretraining_transfer_smoke_v0.csv
```

Initial result:

| Method | Zero-Shot | Adaptation AUC | First Success | Final Score | Transfer Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scratch` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `simple_q_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |

Interpretation:

The first smoke result does not show transfer. That is expected with tiny budgets and a hard target. The important result is that the new evaluation path works end to end:

```text
source pretraining -> artifact -> target adaptation -> transfer report
```

## Multi-Method Smoke Test

Command:

```bash
uv run python scripts/run_pretraining_transfer_smoke.py \
  --source-env MiniGrid-FourRooms-v0 \
  --target-env MiniGrid-MultiRoom-N4-S5-v0 \
  --seed 7 \
  --state-encoder geometry \
  --pretrain-episodes 20 \
  --adapt-episodes 20 \
  --eval-every 10 \
  --eval-episodes 3 \
  --pretrain-methods simple_q_pretrain,operate_replay_pretrain \
  --output outputs/pretraining_transfer_smoke_v1.csv
```

Result:

| Method | Zero-Shot | Adaptation AUC | First Success | Final Score | Transfer Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scratch` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `simple_q_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `operate_replay_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |

Interpretation:

This still does not demonstrate transfer. It does demonstrate that multiple pretraining schedules can now be evaluated by the same report object and CSV output. That is the required foundation before we compare cycle designs instead of maze-specific final scores.

## Expanded Adapter Smoke Test

Command:

```bash
uv run python scripts/run_pretraining_transfer_smoke.py \
  --source-env MiniGrid-FourRooms-v0 \
  --target-env MiniGrid-MultiRoom-N4-S5-v0 \
  --seed 7 \
  --state-encoder geometry \
  --pretrain-episodes 16 \
  --adapt-episodes 12 \
  --eval-every 6 \
  --eval-episodes 2 \
  --output outputs/pretraining_transfer_smoke_v2.csv
```

Result:

| Method | Zero-Shot | Adaptation AUC | First Success | Final Score | Transfer Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scratch` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `simple_q_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `operate_replay_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `cyclic_motif_fast_replay_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |
| `cyclic_subgoal_ecology_replay_pretrain` | `0.0000` | `0.0000` | `-1` | `0.0000` | `0.0000` |

Interpretation:

This smoke is not a performance claim. It confirms that four pretraining methods can now export artifacts and enter the same transfer evaluation.

The runner now also writes artifact diagnostics into the CSV:

```text
artifact_source_score
artifact_source_success
artifact_train_successes
artifact_replay_bank_size
artifact_motif_count
artifact_active_niches
artifact_subgoal_motif_count
artifact_transition_motif_count
artifact_metadata_json
```

This matters because target success is still too sparse. Without artifact diagnostics, every method looks identical. With diagnostics, we can distinguish:

| Method | Artifact Signal In v2 Smoke |
| --- | --- |
| `cyclic_motif_fast_replay_pretrain` | nonzero source success, but empty fast motif bank under this tiny budget |
| `cyclic_subgoal_ecology_replay_pretrain` | zero source success, but nonempty subgoal/transition memory |

That is exactly the distinction the pretraining frame is meant to expose:

```text
Did this method create inherited structure even before transfer success appears?
```

## Transfer Diagnostic V1

The first non-smoke diagnostic uses three seeds and keeps the budget small enough for local iteration:

```bash
uv run python scripts/run_pretraining_transfer_diagnostic.py \
  --seeds 7,17,27 \
  --pretrain-episodes 40 \
  --adapt-episodes 30 \
  --eval-every 10 \
  --eval-episodes 4 \
  --output outputs/pretraining_transfer_diagnostic_v1.csv \
  --summary-output outputs/pretraining_transfer_diagnostic_v1_summary.csv
```

Source:

```text
MiniGrid-FourRooms-v0
```

Target:

```text
MiniGrid-MultiRoom-N4-S5-v0
```

Summary:

| Method | Target Final | Target AUC | Source Score | Source Success | Replay Bank | Motifs | Active Niches | Subgoal Motifs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scratch` | `0.0000` | `0.0000` | - | - | - | - | - | - |
| `simple_q_pretrain` | `0.0000` | `0.0000` | `0.2553` | `0.1111` | - | - | - | - |
| `operate_replay_pretrain` | `0.0000` | `0.0000` | `0.1263` | `0.0556` | `6.6667` | - | - | - |
| `cyclic_motif_fast_replay_pretrain` | `0.0000` | `0.0000` | `0.2487` | `0.1667` | `0.0000` | `0.0000` | `20.0000` | - |
| `cyclic_subgoal_ecology_replay_pretrain` | `0.0000` | `0.0000` | `0.1028` | `0.0556` | `1.0000` | `61.6667` | `15.3333` | `22.0000` |

Interpretation:

No method transfers to MultiRoom under this budget. This is a real negative result, not a failed run.

The artifact layer is informative:

```text
cyclic_motif_fast_replay_pretrain:
  learns some source behavior
  produces active niches
  still creates zero fast motifs

cyclic_subgoal_ecology_replay_pretrain:
  weaker source score
  reliably creates many motifs
  reliably creates subgoal transition motifs
```

This points to the next bottleneck:

```text
artifact creation is no longer the only issue;
artifact transferability and target-side reuse are now the issue.
```

The next experiment should not simply increase budget. It should add an intermediate target or a target-side artifact reuse diagnostic, because FourRooms -> MultiRoom is currently too sparse to distinguish transfer usefulness by final score.

## Decision Rule

After the framework exists, new cycle variants should only be promoted when they improve at least one of:

```text
adaptation_auc
time_to_first_success
final_score_under_budget
negative_transfer_rate
cross_seed_stability
```

Diagnostic metrics may explain why a method works or fails. They should not be used alone to claim a better pretraining method.

## Immediate Next Steps

1. Implement the minimal artifact and transfer report classes.
2. Implement a smoke transfer runner.
3. Establish scratch baseline vs simple pretraining baseline.
4. Then add adapters for existing cycle schedules.
5. Re-evaluate old winners as pretraining schedules, not as final task solvers.

## Short Thesis

The project should now be evaluated by this question:

```text
Does this evolutionary cycle create inherited structure that improves future adaptation?
```

That is the bridge from toy environments to financial rolling learning.
