# Handoff: 2026-04-19 Behavior Prior Transfer

## Purpose

This note is for the next maintainer taking over the project after the
`behavior-prior-transfer` step and the first public GitHub publication.

It records:

- what was implemented;
- what evidence was gathered;
- what was published to GitHub;
- what is still unfinished;
- what to be careful about before the next algorithmic step.

## Current Repo State

Local branch at handoff:

```text
experiment/behavior-prior-transfer
```

Current HEAD:

```text
94e3ad0859544e450fe2ffbd3c893a13e8ea07c9
Add behavior prior transfer scaffold
```

Latest historical tag before this work:

```text
v1.4.6-version-history-and-tests
```

Important branch fact:

```text
main is still the old initial branch
the active research line is not on main
```

Because of that, the public GitHub default branch was set to:

```text
experiment/behavior-prior-transfer
```

## Main Technical Work Completed

This step did not add another large cycle variant.

It focused on the narrower missing layer identified after `v1.4.5`:

```text
target-side traces existed
but trace reinforcement alone was too weak
```

The implemented upgrade was:

```text
raw target trace
-> structural state signature
-> reusable behavior prior
-> early adaptation execution
```

### Main Code Changes

Core new behavior-prior path:

- [`src/minigrid_encoders.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/minigrid_encoders.py)
  Added `StateSignature` for structural matching.
- [`src/minigrid_adapter.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/minigrid_adapter.py)
  Exposed `state_signature` on MiniGrid rollouts.
- [`src/agents.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/agents.py)
  Extended `Rollout` to carry per-step `state_signatures`.
- [`src/core/behavior.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/core/behavior.py)
  Added `BehaviorPrior` and `BehaviorLibrary`.
- [`src/transfer/evaluator.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/transfer/evaluator.py)
  Upgraded target reuse from trace reinforcement only to:
  - motif-to-prior extraction;
  - prior merging;
  - early adaptation prior execution;
  - execution diagnostics;
  - executed-vs-idle episode diagnostics.

Rollout compatibility updates:

- [`src/pretraining/minigrid_schedules.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/pretraining/minigrid_schedules.py)
- [`src/minigrid_train.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/minigrid_train.py)
- [`scripts/run_navigation_motif_diagnostics.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/scripts/run_navigation_motif_diagnostics.py)
- [`scripts/run_minigrid_diagnostics.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/scripts/run_minigrid_diagnostics.py)

Artifact/reporting updates:

- [`src/pretraining/artifacts.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/src/pretraining/artifacts.py)
  Added `behavior_library` slot to `PretrainArtifact`.
- [`scripts/run_pretraining_transfer_diagnostic.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/scripts/run_pretraining_transfer_diagnostic.py)
  Added new target-reuse reporting fields, including:
  - prior support;
  - matched and executed counts;
  - executed steps;
  - executed-episode vs idle-episode structure metrics.

### New Tests

Added:

- [`tests/test_behavior_priors.py`](/Users/qlqwpy/Documents/游乐园/maze_novelty/tests/test_behavior_priors.py)

The tests cover:

- structural signature matching;
- prior-library merge behavior;
- prior extraction from rollouts;
- prior execution during adaptation episodes;
- executed-vs-idle episode statistics bookkeeping.

## Documentation Added Or Updated

New main writeup for this step:

- [`docs/v1.4.7_behavior_prior_transfer.md`](/Users/qlqwpy/Documents/游乐园/maze_novelty/docs/v1.4.7_behavior_prior_transfer.md)

Updated:

- [`docs/pretraining_transfer_framework.md`](/Users/qlqwpy/Documents/游乐园/maze_novelty/docs/pretraining_transfer_framework.md)
- [`docs/version_history.md`](/Users/qlqwpy/Documents/游乐园/maze_novelty/docs/version_history.md)
- [`README.md`](/Users/qlqwpy/Documents/游乐园/maze_novelty/README.md)

The intent was to keep this step aligned with the project's existing version
discipline:

```text
single mechanism focus
small probe
written interpretation
no inflated claims
```

## Evidence Gathered

Probe outputs were written locally and were not committed:

- `outputs/pretraining_behavior_prior_probe.csv`
- `outputs/pretraining_behavior_prior_probe_summary.csv`

Command used:

```bash
uv run python scripts/run_pretraining_transfer_diagnostic.py \
  --seeds 7 \
  --target-envs MiniGrid-MultiRoom-N2-S4-v0,MiniGrid-MultiRoom-N4-S5-v0 \
  --pretrain-episodes 20 \
  --adapt-episodes 20 \
  --eval-every 10 \
  --eval-episodes 3 \
  --include-target-reuse \
  --output outputs/pretraining_behavior_prior_probe.csv \
  --summary-output outputs/pretraining_behavior_prior_probe_summary.csv
```

### Honest Result

Primary transfer remained:

```text
final_score = 0
adaptation_auc = 0
```

No transfer breakthrough should be claimed from this step.

### Useful Positive Signal

The new behavior-prior mechanism is not silent anymore.

For example, on `MiniGrid-MultiRoom-N4-S5-v0`:

- `operate_replay_pretrain+target_reuse`
  - `matched_prior_count = 368`
  - `executed_prior_count = 104`
  - `executed_prior_steps = 624`
  - `executed_episode_avg_region_transitions = 13.0`
- `cyclic_motif_fast_replay_pretrain+target_reuse`
  - `best_prior_support = 8`
  - `executed_episode_avg_region_transitions = 7.875`
  - `executed_episode_avg_mobility = 0.1826`
- `cyclic_subgoal_ecology_replay_pretrain+target_reuse`
  - `executed_episode_avg_subgoal_score = 0.1595`
  - `executed_episode_avg_region_transitions = 11.875`

Interpretation at handoff:

- `motif_fast` currently looks best for repeated short-prior consistency;
- `operate_replay` currently looks strongest for transition-driving execution;
- `subgoal_ecology` currently looks strongest for progress-oriented execution.

This is not yet enough to promote a winner. It is enough to justify continuing
on the same line instead of adding another large cycle variant.

## Checks Completed

The following checks passed after the latest code changes:

```bash
uv run ruff check src scripts tests
uv run pyright src scripts
uv run pytest
```

At the time of handoff, the local worktree is clean.

## GitHub Publication Details

Public repository created:

- [wang-yzh/maze-novelty](https://github.com/wang-yzh/maze-novelty)

Repository details at handoff:

```text
name: wang-yzh/maze-novelty
visibility: public
default branch: experiment/behavior-prior-transfer
url: https://github.com/wang-yzh/maze-novelty
description: Cyclic novelty, transfer, and behavior prior experiments in GridWorld and MiniGrid.
```

Remote configured locally:

```text
origin https://github.com/wang-yzh/maze-novelty.git
```

### Publication Operations Performed

1. Re-authenticated GitHub CLI as `wang-yzh`.
2. Set local repo commit identity:
   - name: `wang-yzh`
   - email: `arnaud0.0lw@gmail.com`
3. Committed the current work:

   ```text
   94e3ad0 Add behavior prior transfer scaffold
   ```

4. Created the public GitHub repository.
5. Pushed:
   - current branch;
   - all local branches;
   - all historical tags.
6. Set the GitHub default branch to:

   ```text
   experiment/behavior-prior-transfer
   ```

### Branches Published

Published branches include:

- `experiment/behavior-prior-transfer`
- `experiment/framework-upgrade`
- `experiment/explore-operate-cycle`
- `experiment/minigrid-benchmark`
- `main`

### Tags Published

All historical tags through `v1.4.6-version-history-and-tests` were pushed.

Important note:

```text
the current behavior-prior-transfer commit is committed and published
but not yet tagged
```

This is deliberate. It should only be tagged after the team decides this step is
stable enough to freeze as the next version point.

## Suggested Immediate Next Step

Do not add a new cycle variant yet.

The cleanest next continuation is:

1. run the same behavior-prior mechanism on 3 seeds;
2. confirm whether the executed-episode structure differences are stable;
3. only then decide whether to:
   - tune the prior executor,
   - tune prior extraction quality thresholds,
   - or promote the step into a new tag.

## Things The Next Maintainer Should Not Accidentally Break

- Do not move default development back to `main` without an intentional branch
  cleanup plan.
- Do not interpret `target_probe_*` fields as post-adaptation behavior. Those
  are pre-reuse probe measurements.
- Do not treat prior count alone as progress.
- Do not commit generated `outputs/`.
- Do not claim success before `adaptation_auc`, `time_to_first_success`, or
  final score actually improve under a larger run.

## Short Summary

At handoff, the project is in this state:

```text
public GitHub repo exists
current branch is published
behavior-prior scaffold is implemented
the mechanism executes
the transfer problem is still unsolved
the next maintainer should validate executed-episode structure across more seeds
before changing the algorithm family
```
