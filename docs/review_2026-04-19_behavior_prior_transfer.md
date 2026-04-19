# Review: 2026-04-19 Behavior Prior Transfer

## Scope

This review covers the behavior-prior transfer work on branch:

```text
experiment/behavior-prior-transfer
```

Reviewed diff against:

```text
experiment/framework-upgrade
```

Primary inputs:

- `docs/handoff_2026-04-19_behavior_prior_transfer.md`
- `src/transfer/evaluator.py`
- `src/core/behavior.py`
- `src/minigrid_encoders.py`
- `src/minigrid_adapter.py`
- `src/pretraining/minigrid_schedules.py`
- `tests/test_behavior_priors.py`

## Review Outcome

Overall judgment:

```text
The branch is coherent and publishable as an experimental scaffold.
The new mechanism is real, the tests are meaningful, and the reporting path is much stronger.
The branch still carries one important open algorithmic risk.
```

## Fixed During Review

### 1. Best-Artifact Selection Bug

Status:

```text
fixed in this review
```

Issue:

- `PretrainArtifact.best_agent()` returns `hall_of_fame[0]`.
- In the cyclic pretraining builders, the explicitly re-evaluated `best` agent was
  appended at the end of `hall_of_fame`.
- That means transfer evaluation could use an older elite instead of the
  strongest evaluated artifact.

Impact:

```text
Transfer comparisons could be quietly downgraded by using the wrong pretrained agent.
```

Fix applied:

- reordered cyclic artifact construction so the evaluated `best` agent is stored
  first in `hall_of_fame`.

Files:

- `src/pretraining/minigrid_schedules.py`

## Open Findings

### [P2] Structural prior matching is probably stricter than the action semantics require

Files:

- `src/minigrid_encoders.py`
- `src/core/behavior.py`
- `src/transfer/evaluator.py`

Observation:

- `StateSignature.matches()` currently requires exact `direction` equality.
- But the local spatial pattern, topology, goal-bin, and action traces are
  defined in an egocentric frame.
- Left/right/forward priors are therefore relative to the agent's current view,
  not to the global world direction.

Why this matters:

```text
Two structurally identical local situations can fail to match only because the
agent is facing a different absolute direction in the map.
```

Likely effect:

- lower prior support counts;
- fewer matches during adaptation;
- weaker execution reuse than the mechanism might otherwise achieve.

Why this was not changed in the review:

```text
This is an algorithmic choice, not a mechanical bug.
Changing it would alter the behavior-prior hypothesis itself and should be
validated by an explicit experiment.
```

Recommendation:

- run an A/B comparison with:
  - current `direction`-strict matching;
  - direction-agnostic matching;
- compare:
  - matched prior count;
  - executed prior count;
  - executed episode region transitions;
  - adaptation AUC.

### [P3] `behavior_library` is now part of `PretrainArtifact`, but the current transfer path still rebuilds priors from target probes

Files:

- `src/pretraining/artifacts.py`
- `src/transfer/evaluator.py`

Observation:

- the artifact can now carry a `behavior_library`;
- the current behavior-prior transfer path does not consume that field;
- instead it constructs priors entirely from target-side probe rollouts.

Why this matters:

```text
The repository now has two concepts:
artifact-carried behavior structure
and target-probe-derived behavior structure

but only the second one is active in transfer.
```

This is not wrong, but it should be made explicit in future design work.

## Checks Run

The following checks passed on the reviewed branch:

```bash
uv run ruff check src scripts tests
uv run pyright src scripts
uv run pytest
```

## Review Conclusion

My conclusion after reading the branch and running the checks is:

```text
This is a legitimate next step after target-reuse trace replay.
It does not oversell the results.
It improves the experimental instrumentation substantially.
The branch is suitable to publish as a public research checkpoint.
```

What I would do next:

1. keep this branch as a public checkpoint;
2. avoid another large cycle variant;
3. test the matching rule explicitly;
4. decide whether the long-term path is:
   - target-side option execution;
   - behavior priors as an early adaptation policy prior;
   - or artifact-side prior transfer instead of probe-side reconstruction.

