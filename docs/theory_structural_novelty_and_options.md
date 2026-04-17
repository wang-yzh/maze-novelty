# Structural Novelty And Options

## Why This Document Exists

The project has reached a real bottleneck.

The v1.2 MultiRoom diagnostic showed:

- `cyclic_operate_replay`, `cyclic_motif_fast_replay`, and `cyclic_subgoal_ecology_replay` all had `primary_score = 0`.
- `cyclic_subgoal_ecology_replay` filled its motif bank, proving that failed trajectories can become memory.
- That memory did not become goal-reaching behavior.
- Simple wall-following heuristics produced much higher mobility and room-transition counts than the learned policies.

This means the bottleneck is not another phase schedule. The bottleneck is the missing layer between local actions and reusable behavior.

The next theory target is:

```text
How can novelty produce behavior structures that are reusable, composable, and robust?
```

## Current Architecture

The current system has three layers.

```text
Cycle layer:
  explore / operate / stress / bottleneck / replay

Memory layer:
  novelty archive / niche archive / replay bank / motif bank / subgoal memory

Action layer:
  tabular Q-table over primitive actions
```

The cycle and memory layers are now rich enough to generate useful diagnostics. The weak layer is the action layer.

The action layer still thinks in primitive actions:

```text
left
right
forward
```

MultiRoom requires behavior chunks:

```text
follow wall
leave corner
cross doorway
continue through a room
avoid local revisit loops
```

The system can store trajectories, but it does not yet compress them into reusable behavior chunks. That is why subgoal memory fills but success remains zero.

## Key Theoretical Claim

Novelty is not enough.

The useful target is structural novelty:

```text
novelty that creates reusable structure
```

A behavior is structurally novel when it satisfies at least one of these:

- It connects two previously separate regions of behavior space.
- It repeatedly appears inside high-progress failed trajectories.
- It survives stress or evaluation seed changes.
- It can be replayed as a short action program.
- It improves later exploration by reducing local search cost.

This is different from raw novelty:

```text
raw novelty:
  "This trajectory is different."

structural novelty:
  "This trajectory contains a reusable transition."
```

## Useful Theory Anchors

These are not direct ancestors of our algorithm, but they are useful nearby ideas.

### Options And Skills

Options are temporally extended actions. Instead of choosing one primitive action at every step, the agent can choose a policy fragment that runs for multiple steps.

For this project, options should stay small and discrete:

```text
forward_run
wall_follow_left
wall_follow_right
doorway_cross
unstuck_turnaround
```

The important part is not the exact option list. The important part is that the system starts treating repeated action structures as inheritable objects.

### Subgoal Discovery

Current `subgoal_score` is diagnostic. It says that a failed rollout made some progress.

A real subgoal should identify a reusable transition:

```text
entered a new region
crossed a bottleneck-like boundary
escaped a revisit loop
reached a high-connectivity point
increased reachable future space
```

Subgoals should not be defined as "closer to the final goal" only. In sparse tasks, the final goal may not be discovered early enough.

### Quality-Diversity

Quality-Diversity helps preserve different kinds of behavior, but it does not automatically make those behaviors composable.

Our current niche archives preserve diversity. The missing step is to mine the preserved behaviors for reusable fragments.

### Go-Explore

Go-Explore separates returning to an interesting state from exploring onward. Our current system has a weaker equivalent through archive and replay, but it does not reliably restore useful high-level states in MultiRoom.

The transferable lesson is:

```text
exploration needs reliable return paths or reusable transition skills
```

### Bottleneck States

In mazes, doors and corridor mouths are bottlenecks. In markets, regime transitions can play a similar role.

A bottleneck is valuable because many useful trajectories pass through it. This suggests that motif extraction should prefer fragments that:

- occur near region changes;
- reduce repeated local motion;
- increase future reachable area;
- appear across seeds or environment variants.

## Revised Model

The model should no longer be described as just a cycle algorithm.

Better description:

```text
Open exploration discovers behavior variation.
Ecological pressure filters brittle variation.
Memory stores reusable traces.
Options compress traces into reusable behavior.
Replay turns reusable behavior into inherited structure.
```

The new stack:

```text
Cycle layer:
  decides when to explore, operate, stress, bottleneck, replay

Ecology layer:
  preserves diverse elites and applies pressure

Structural memory layer:
  stores motifs, subgoals, bottlenecks, and transition fragments

Option layer:
  converts repeated fragments into short executable behavior programs

Primitive policy layer:
  tabular Q over local observations and actions
```

## Why v1.2 Did Not Solve MultiRoom

The v1.2 subgoal memory did solve one problem:

```text
sparse success no longer means empty memory
```

But it did not solve the next problem:

```text
stored memory was not executable enough
```

Most stored subgoal motifs are still local action traces. They do not know whether they are:

- wall-following;
- door-crossing;
- loop escape;
- region transfer;
- random wandering.

So replay can strengthen fragments, but it cannot yet prefer the fragments that behave like navigation primitives.

The heuristic comparison makes this clear:

```text
wall-following:
  higher transitions
  higher mobility
  still no success

learned policies:
  lower transitions
  lower mobility
  many motifs but weak navigational structure
```

Therefore the next layer must classify and extract navigation-like motifs.

## Navigation Motif V1

The next coding target should be `navigation_motif_v1`.

It should not try to solve MultiRoom immediately. It should answer one question:

```text
Can learned/extracted motifs beat simple wall-following on navigation diagnostics?
```

### Motif Types

Minimum motif classes:

```text
forward_run:
  repeated forward movement with low collision rate

wall_follow_left:
  pattern that turns left after blocked movement and then advances

wall_follow_right:
  pattern that turns right after blocked movement and then advances

region_transition:
  action segment around a region change

unstuck:
  segment that reduces revisit ratio after a local loop
```

These are implementation labels, not hard-coded final strategies.

### Extraction Signals

Use trajectory-only signals first:

```text
positions
actions
region sequence
mobility
revisit ratio
collision proxy: same position after forward action
region transition count
```

Avoid relying on MiniGrid object-specific details in v1. This keeps the mechanism more general.

### Motif Quality

A navigation motif should score high when it:

```text
increases mobility
reduces repeated positions
crosses region boundaries
keeps action length short enough to replay
appears across multiple rollouts
```

Initial quality:

```text
navigation_quality =
  0.30 * mobility
+ 0.25 * region_transition
+ 0.20 * revisit_reduction
+ 0.15 * compactness
+ 0.10 * recurrence
```

### Replay Rule

Navigation motifs should not be replayed exactly like success motifs.

Replay priority:

```text
confirmed_success motifs:
  highest reward

navigation_transition motifs:
  medium reward

wall_follow / unstuck motifs:
  medium-low reward

generic subgoal motifs:
  low reward
```

The purpose is not to force a single wall-following policy. The purpose is to make the population inherit movement competence.

## Experimental Gates

Do not judge `navigation_motif_v1` by success first.

Gate 1:

```text
Does it increase mobility above cyclic_subgoal_ecology_replay?
```

Gate 2:

```text
Does it approach or beat wall-following transition counts?
```

Gate 3:

```text
Does it increase best_new_regions or best_region_transitions across 3 seeds?
```

Gate 4:

```text
Only after Gates 1-3, check primary success.
```

If it cannot pass Gates 1-3, success is unlikely to improve by tuning phase schedules.

## Generalization Beyond Maze

The idea should not be "add wall-following to mazes."

The general idea is:

```text
extract executable transition motifs from failed high-progress behavior
```

Maze analogues:

```text
doorway crossing
room transition
unstuck behavior
```

Market analogues:

```text
regime transition handling
drawdown recovery behavior
position de-risking motif
low-turnover stable exposure motif
```

The shared abstraction:

```text
local behavior fragment that improves future reachable opportunity
```

## What Not To Do Next

Do not add more global cycle variants yet.

Avoid:

- another ecology phase;
- another replay timing pattern;
- more population hyperparameter tuning;
- declaring success from diagnostic score alone;
- using financial profit tests as the next algorithm debugger.

Those can come later. The current missing piece is behavior structure.

## Next Coding Plan

Implement in this order:

1. Add a `navigation_motifs.py` module.
2. Extract candidate motifs from rollout actions and positions.
3. Add metrics:
   - `navigation_motif_count`
   - `region_transition_motif_count`
   - `wall_follow_motif_count`
   - `unstuck_motif_count`
   - `collision_proxy_rate`
4. Add a diagnostic-only runner that extracts motifs from existing rollouts without changing training.
5. Add `cyclic_navigation_motif_replay` only after extraction metrics look sane.
6. Compare against wall-following diagnostics before checking primary success.

## Short Thesis

The project should now be framed this way:

```text
Novelty creates variation.
Pressure selects viable variation.
Memory stores useful traces.
Options compress traces into reusable behavior.
Replay makes behavior inheritable.
```

The next breakthrough is not a better schedule. It is turning novelty from "different trajectories" into "new reusable behavioral structure."
