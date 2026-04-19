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

## Transfer Ladder Probe V1

The diagnostic runner now supports target ladders:

```text
--target-envs MiniGrid-MultiRoom-N2-S4-v0,MiniGrid-MultiRoom-N4-S5-v0
```

It also reports target-side zero-shot behavior diagnostics before adaptation:

```text
target_probe_subgoal_score
target_probe_region_transitions
target_probe_new_regions
target_probe_mobility
target_probe_success
```

Command:

```bash
uv run python scripts/run_pretraining_transfer_diagnostic.py \
  --seeds 7,17,27 \
  --target-envs MiniGrid-MultiRoom-N2-S4-v0,MiniGrid-MultiRoom-N4-S5-v0 \
  --pretrain-episodes 40 \
  --adapt-episodes 30 \
  --eval-every 10 \
  --eval-episodes 4 \
  --output outputs/pretraining_transfer_ladder_probe_v1.csv \
  --summary-output outputs/pretraining_transfer_ladder_probe_v1_summary.csv
```

Final target success remains zero on both targets. The useful signal is in the target probe:

| Target | Method | Target Probe Subgoal | Region Transitions | New Regions | Mobility |
| --- | --- | ---: | ---: | ---: | ---: |
| `MultiRoom-N2-S4` | `simple_q_pretrain` | `0.0111` | `0.4167` | `1.3333` | `0.0062` |
| `MultiRoom-N2-S4` | `operate_replay_pretrain` | `0.0224` | `2.2500` | `1.1667` | `0.0609` |
| `MultiRoom-N2-S4` | `cyclic_motif_fast_replay_pretrain` | `0.0616` | `3.2500` | `1.5000` | `0.0534` |
| `MultiRoom-N2-S4` | `cyclic_subgoal_ecology_replay_pretrain` | `0.0186` | `1.5000` | `1.0833` | `0.0583` |
| `MultiRoom-N4-S5` | `simple_q_pretrain` | `0.0160` | `0.5833` | `1.2500` | `0.0173` |
| `MultiRoom-N4-S5` | `operate_replay_pretrain` | `0.0230` | `1.8333` | `1.1667` | `0.0618` |
| `MultiRoom-N4-S5` | `cyclic_motif_fast_replay_pretrain` | `0.0607` | `2.2500` | `1.5000` | `0.0371` |
| `MultiRoom-N4-S5` | `cyclic_subgoal_ecology_replay_pretrain` | `0.0680` | `3.0000` | `1.6667` | `0.0654` |

Interpretation:

```text
target success is still too sparse;
target-side behavior is not identical across pretraining artifacts.
```

The ladder reveals a useful split:

```text
N2-S4:
  motif-fast has the best target probe subgoal score and region transitions.

N4-S5:
  subgoal-ecology has the best target probe subgoal score,
  most region transitions,
  most new regions,
  and highest mobility.
```

This is the first evidence in the transfer framework that artifact type affects target-side behavior before success appears. It is not yet proof of transfer performance, but it tells us the next coding target:

```text
target-side artifact reuse must become an adaptation mechanism,
not just a diagnostic.
```

## Target Reuse V1

Target-side artifact reuse has been implemented as an adaptation option.

Mechanism:

```text
1. Clone the pretraining artifact's best agent.
2. Probe it on the target env with small epsilon.
3. Keep target probe rollouts with useful subgoal, mobility, region-transition, or success signal.
4. Convert those rollouts into target reuse items.
5. During early target adaptation, reinforce those target-side traces before normal Q-learning episodes.
```

The diagnostic runner enables this with:

```text
--include-target-reuse
```

Command:

```bash
uv run python scripts/run_pretraining_transfer_diagnostic.py \
  --seeds 7,17,27 \
  --target-envs MiniGrid-MultiRoom-N2-S4-v0,MiniGrid-MultiRoom-N4-S5-v0 \
  --pretrain-episodes 40 \
  --adapt-episodes 30 \
  --eval-every 10 \
  --eval-episodes 4 \
  --include-target-reuse \
  --output outputs/pretraining_transfer_reuse_v1.csv \
  --summary-output outputs/pretraining_transfer_reuse_v1_summary.csv
```

Engineering fix:

```text
QAgent.clone no longer shares the same RNG object.
PretrainArtifact.best_agent now returns a deterministic seeded clone.
```

This matters because ordinary transfer and target-reuse transfer must not perturb each other through shared random tie-breaking.

Result:

| Target | Method | Final | AUC | Reuse Items | Avg Reuse Score | Best Reuse Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `N2-S4` | `simple_q_pretrain+target_reuse` | `0.0000` | `0.0000` | `2.6667` | `0.2230` | `0.3125` |
| `N2-S4` | `operate_replay_pretrain+target_reuse` | `0.0000` | `0.0000` | `5.0000` | `0.2549` | `0.4156` |
| `N2-S4` | `cyclic_motif_fast_replay_pretrain+target_reuse` | `0.0000` | `0.0000` | `5.3333` | `0.1818` | `0.4315` |
| `N2-S4` | `cyclic_subgoal_ecology_replay_pretrain+target_reuse` | `0.0000` | `0.0000` | `5.0000` | `0.2978` | `0.5208` |
| `N4-S5` | `simple_q_pretrain+target_reuse` | `0.0000` | `0.0000` | `4.0000` | `0.2902` | `0.4512` |
| `N4-S5` | `operate_replay_pretrain+target_reuse` | `0.0000` | `0.0000` | `5.0000` | `0.1802` | `0.4147` |
| `N4-S5` | `cyclic_motif_fast_replay_pretrain+target_reuse` | `0.0000` | `0.0000` | `5.6667` | `0.3177` | `0.5375` |
| `N4-S5` | `cyclic_subgoal_ecology_replay_pretrain+target_reuse` | `0.0000` | `0.0000` | `5.6667` | `0.2028` | `0.4647` |

Interpretation:

This is another useful negative result:

```text
target reuse items exist;
they have measurable target-side structure;
this simple trace reinforcement still does not convert them into target success.
```

So the bottleneck has moved one step deeper:

```text
not "can we find target-side reusable traces?"
but "can we compose or execute them as reusable skills/options?"
```

The next version should not just increase trace replay reward. It should test whether target reuse should become:

```text
short option execution,
subgoal-conditioned exploration,
or a temporary behavior prior during early adaptation.
```

## Behavior Prior Transfer Probe

The next minimal probe has now been implemented on the current branch:

```text
state signature
-> navigation-motif prior extraction
-> repeated-prior merging
-> early adaptation prior execution
```

This is intentionally still smaller than a full options system. The important
question is not "did transfer succeed already?" but:

```text
did the reusable-structure executor actually fire?
```

The first quick probe remained all-zero on final transfer score. That negative
result should be preserved.

The useful positive signal is different:

- repeated structural priors were extracted;
- prior support could exceed `1`, so duplicated fragments no longer remained
  isolated raw traces;
- matched-prior counts and executed-prior counts were both substantial during
  early adaptation;
- executed-prior episodes already showed different structural profiles across
  artifact families, even though final score remained zero;
- therefore the bottleneck moved again:

```text
not "can we execute target-side priors?"
but "how should executed priors change adaptation dynamics enough to create success?"
```

The dedicated writeup for this step is:

- `docs/v1.4.7_behavior_prior_transfer.md`

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
