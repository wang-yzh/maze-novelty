# Research Roadmap

## Purpose

This document defines the project's real target and the next research path.

It is not a benchmark note.
It is not a feature wishlist.
It is a control document for answering one question:

```text
Are we building a pretraining algorithm that can accumulate reusable structure,
transfer it, and keep improving across tasks?
```

## The Real Goal

The real goal is not:

```text
solve a maze
beat a baseline on one benchmark
invent a more complex training cycle
```

The real goal is:

```text
build an evolutionary pretraining system that:
1. generates diverse behaviors,
2. compresses useful behaviors into inherited structure,
3. transfers that structure into new tasks,
4. adapts under limited budget,
5. and keeps improving by feeding new structure back into the system.
```

Long term, this should point toward rolling strategy learning in changing
domains such as finance.

## Current Position

We are not at the final algorithm.

We are at the stage where we have identified several necessary ingredients and
started testing whether they are real.

The central progress so far is:

```text
the project is no longer "maze solving"
the project is now "pretraining artifact -> transfer -> adaptation"
```

That is the correct framing for the long-term objective.

## What We Have Actually Confirmed

### 1. Cycle history changes later behavior

Different pretraining schedules produce different target-side behavior, even
when final target success remains zero.

Meaning:

```text
the artifact is not empty
the system carries historical structure into the new task
```

### 2. Success-only memory is not enough

`cyclic_motif_fast_replay` is powerful when fast successes exist, but its memory
can stay empty in sparse-success settings.

Meaning:

```text
fast-success replay is a strong branch
but it cannot be the whole theory
```

### 3. Failure-derived memory is real

`cyclic_subgoal_ecology_replay` reliably creates subgoal and transition memory
from failed but high-progress behavior.

Meaning:

```text
useful inherited structure does not need to come only from success
```

### 4. Target-side reusable traces exist

Target-side probes and target-reuse diagnostics showed that new-task rollouts
contain reusable local structure.

Meaning:

```text
the problem is no longer "is there anything reusable at all?"
```

### 5. Simple replay is too weak

Trace replay and early target-side reinforcement did not convert that structure
into nonzero target success.

Meaning:

```text
finding structure is not enough
storing structure is not enough
we need a stronger way to use structure
```

## What Is Still Unproven

These are the things we do **not** know yet.

### 1. We have not proven true transfer performance

Current hard transfer tasks still return:

```text
final_score = 0
adaptation_auc = 0
```

So we cannot claim practical transfer yet.

### 2. We have not proven continual evolution

We do not yet have a stable loop of:

```text
pretrain -> transfer -> adapt -> extract new reusable structure -> re-pretrain
```

That loop is the real threshold for "keeps improving."

### 3. We have not proven domain-general structure

The current project is still MiniGrid-heavy.

That is acceptable at this stage, but it means:

```text
we are testing the outer learning logic
not yet demonstrating broad-domain generalization
```

## Core Thesis Going Forward

The project should now be treated as a search for this algorithmic shape:

```text
open exploration
-> structural memory
-> transferable priors or options
-> target-side execution and adaptation
-> feedback of new structure into future pretraining
```

This is the path from novelty search to real pretraining.

## What We Should Not Do Next

To avoid getting lost again, these are explicit anti-goals.

### Do not add another large cycle variant now

We already have enough cycle families to learn from:

- `cyclic_operate_replay`
- `cyclic_motif_fast_replay`
- `cyclic_subgoal_ecology_replay`

Another large cycle now would increase complexity faster than understanding.

### Do not optimize only for final maze success

Hard tasks are too sparse for final score to be the only judge.

### Do not treat more motifs or more niches as proof of progress

Memory volume is not memory usefulness.

### Do not jump into finance directly

Finance is still too complex for the current mechanism maturity.

We need an intermediate stage where transfer, reuse, and continual structure
update become reliable.

## The Three Most Important Open Questions

### Question 1

```text
How should reusable target-side structure be executed?
```

This is now the main question.

Possible answers:

- short option execution;
- behavior prior during early adaptation;
- subgoal-conditioned exploration;
- hybrid replay + execution.

### Question 2

```text
What matching rule best identifies "the same reusable situation"?
```

Current `StateSignature` matching may be too strict, especially on direction.

This is a smaller but important question because poor matching can hide the true
value of priors.

### Question 3

```text
How should new-task structure flow back into the pretraining system?
```

This is the bridge to continual evolution.

Without it, the system is only doing one-way transfer.

## Recommended Research Loop

For the next stage, the project should use a smaller, more disciplined loop:

```text
1. pretrain artifact
2. target probe
3. extract reusable target-side structure
4. execute or condition on that structure during early adaptation
5. measure adaptation and structural behavior
6. feed validated structure back into the artifact model
```

## Recommended Method Set

We should narrow the core comparison set to three historical families:

### `cyclic_operate_replay`

Role:

```text
clean control baseline
```

Use it to test whether new reuse mechanisms help beyond a simple strong cycle.

### `cyclic_motif_fast_replay`

Role:

```text
speed-disciplined success memory baseline
```

Use it when the environment has enough success signal.

### `cyclic_subgoal_ecology_replay`

Role:

```text
failure-memory and sparse-signal baseline
```

Use it when success is rare and structural progress matters more than direct
completion.

## Near-Term Development Sequence

This is the sequence I recommend now.

### Stage A: Match Better

Target:

```text
improve or ablate StateSignature matching
```

Specific experiment:

- strict matching vs direction-agnostic matching;
- same targets;
- same three core methods.

Primary readouts:

- matched prior count;
- executed prior count;
- executed episode region transitions;
- executed episode subgoal score;
- adaptation AUC.

Goal:

```text
find out whether useful priors are currently being blocked by overly strict matching
```

### Stage B: Use Structure Better

Target:

```text
upgrade target reuse from trace replay to short option-like execution
```

Possible minimal version:

- execute a short prior as a bounded option;
- allow abort if the local structural match collapses;
- compare against open-loop trace execution.

Goal:

```text
test whether reusable structure needs execution semantics, not just replay
```

### Stage C: Feed Structure Back

Target:

```text
close the loop between target adaptation and future pretraining
```

Minimal version:

- collect high-support target-side priors;
- write them back into the artifact;
- pretrain-transfer again on the next target rung.

Goal:

```text
turn one-way transfer into iterative structural accumulation
```

## Recommended Task Ladder

We need a more informative ladder than one hard target.

Suggested order:

1. `MiniGrid-FourRooms-v0`
2. `MiniGrid-MultiRoom-N2-S4-v0`
3. `MiniGrid-MultiRoom-N4-S5-v0`

Later, only after structure reuse becomes useful:

4. another structurally different MiniGrid family
5. non-maze tabular control task
6. domain-specific simulated market environment

The rule is:

```text
do not move to the next domain until structure reuse shows real signal in the current one
```

## Promotion Criteria

A new mechanism should only be promoted if it improves at least one of:

- `adaptation_auc`
- `time_to_first_success`
- `time_to_threshold`
- `target_probe_region_transitions`
- `target_probe_subgoal_score`
- repeated-prior support under fixed budget
- cross-seed stability

And if it makes the system harder to understand, it should clear a higher bar.

## Current Best Interpretation

If I state our current status in one sentence:

```text
we have evidence that pretraining history creates transferable behavioral structure,
but we do not yet have the right execution mechanism to turn that structure into reliable success
```

That means the project is not lost.

It means the project has arrived at its real research problem.

## Immediate Recommendation

If we want to stay disciplined, the next serious version should be:

```text
matching-rule ablation
```

Not a new big cycle.
Not a bigger benchmark.
Not finance yet.

After that:

```text
short option execution
```

After that:

```text
artifact-side structural feedback
```

That is the path most likely to lead from "interesting mechanism" to "real
pretraining algorithm."

