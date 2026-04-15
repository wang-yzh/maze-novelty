# Cyclic Novelty GridWorld Experiment Report

## Summary

This project tests a cyclic novelty-to-exploitation training idea in a small
GridWorld setting. The central hypothesis is:

> Novelty should be used to generate diverse candidate behaviors, but a separate
> exploitation phase must preserve and compress successful behaviors into fast,
> reliable policies.

The current implementation is intentionally simple. It uses tabular Q-learning,
population mutation/crossover, a novelty archive, and a hall-of-fame elite bank.
It does not use neural networks.

The strongest result so far is on a harder 16x16 random-maze benchmark:

- `cyclic_novelty` achieved nonzero test success in 8/10 seeds.
- `q_learning`, `novelty_q`, and `genetic_q` achieved nonzero test success in
  0/10 seeds.
- Average hard-environment `test_score`:
  - `cyclic_novelty`: `0.3010 +/- 0.0827`
  - `genetic_q`: `0.1340 +/- 0.0080`
  - `novelty_q`: `0.0710 +/- 0.0416`
  - `q_learning`: `0.0530 +/- 0.0400`

This is not yet a proof of a general algorithm. It is an early positive signal
in a controlled toy environment.

## Environment

The environment is a square GridWorld maze.

- Start: top-left cell `(0, 0)`.
- Goal: bottom-right cell `(size - 1, size - 1)`.
- Actions: up, down, left, right.
- Obstacles: randomly generated, while ensuring at least one path exists.
- Episode limit: `size * size` steps.
- Base step reward:
  - Each step: `-0.01`
  - Hitting a wall: additional `-0.04`
  - Reaching the goal: `+1.0`

The agent does not receive the full maze layout. Its tabular state is a compact
local observation:

- Four wall bits: whether up, down, left, right are blocked.
- Coarse goal direction: relative row direction and column direction, each in
  `{less, equal, greater}`.

Total observation states:

```text
16 wall patterns * 9 goal-direction patterns = 144 states
```

The archive and visualizations still use true `(row, col)` positions.

## Compared Methods

### 1. `q_learning`

Single tabular Q-learning agent trained with the base task reward.

### 2. `novelty_q`

Single tabular Q-learning agent trained with the base reward plus a novelty
bonus for visiting less-visited true positions.

### 3. `genetic_q`

A population of Q-tables. Each generation:

1. Train each Q-table with task reward.
2. Evaluate each individual.
3. Select elites.
4. Produce the next population via crossover and mutation.

### 4. `cyclic_novelty`

The proposed cyclic method. It alternates between:

1. Novelty cultivation phase.
2. Exploitation phase.

It also keeps a hall-of-fame elite bank so useful policies are not lost during
later novelty mutation.

## Proposed Cyclic Method

### Novelty Cultivation Phase

Purpose:

- Expand behavior diversity.
- Discover new trajectory structures.
- Keep candidate policies from collapsing too early.

Mechanisms:

- Reward includes a novelty bonus based on true-position archive visitation.
- Novelty phase has an increasing time penalty.
- Novelty phase ends when either:
  - no positive gated novelty score appears for `novelty_patience` generations,
    or
  - novelty phase reaches `novelty_max_age = 10`.

Current novelty bonus:

```text
position_bonus(position) = 1 / sqrt(1 + visits(position))
```

Current novelty bonus weight inside cyclic novelty:

```text
bonus_weight = 0.04
```

### Exploitation Phase

Purpose:

- Convert useful exploration into successful, fast policies.
- Select for success and speed rather than novelty alone.

Mechanisms:

- More task-reward Q-learning updates are applied.
- Evaluation uses a success-gated exploitation score.
- Low mutation is used during exploitation.
- A hall-of-fame elite bank is updated and reinserted into the population.
- Exploitation runs for at least `exploit_min_age = 8` generations.

Current exploitation score:

```text
speed = 1 - min(avg_steps, max_steps) / max_steps
success_gate = success_rate^2

exploitation_score =
    0.62 * success_gate
  + 0.28 * speed
  + 0.10 * stability
```

The squared success term strongly favors policies that succeed across multiple
evaluation mazes.

## Final Evaluation Score

All methods are evaluated with the same `test_score`:

```text
speed = 1 - min(avg_steps, max_steps) / max_steps

test_score =
    0.55 * success_rate
  + 0.30 * speed
  + 0.15 * stability
```

Where:

- `success_rate`: fraction of test mazes solved.
- `avg_steps`: average steps among successful episodes. If no successes, this is
  set to `max_steps`.
- `speed`: normalized speed score.
- `stability`: path consistency across test mazes.

Important caveat: `stability` currently rewards repeated path signatures, which
can make a consistently failing policy look stable. This is why `success_rate`
and `test_score` should be interpreted together.

## Archive Metrics

The project records three archive-related metrics:

### `archive_coverage`

Fraction of grid cells ever visited by archived trajectories.

```text
visited_cells / total_cells
```

This metric is easy to saturate and is not sufficient by itself.

### `archive_unique_ratio`

Fraction of archived trajectories that are unique.

```text
unique_trajectories / total_archived_trajectories
```

This helps distinguish repeated behavior from broad exploration.

### `success_path_diversity`

Fraction of successful archived trajectories that are unique.

```text
unique_success_trajectories / total_success_trajectories
```

This tries to capture whether the method finds multiple distinct successful
routes, not just many failed paths.

## Experiment Configurations

### Default Configuration

File: `configs/default.toml`

```toml
size = 12
train_mazes = 18
test_mazes = 8
generations = 45
population = 18
episodes_per_agent = 5
eval_every = 3
novelty_patience = 5
obstacle_prob = 0.22
```

Seeds:

```text
7, 17, 27, 37, 47, 57, 67, 77, 87, 97
```

### Hard Configuration

File: `configs/hard.toml`

```toml
size = 16
train_mazes = 30
test_mazes = 16
generations = 80
population = 24
episodes_per_agent = 6
eval_every = 5
novelty_patience = 5
obstacle_prob = 0.28
```

Seeds:

```text
7, 17, 27, 37, 47, 57, 67, 77, 87, 97
```

## Default 12x12 Results

Final averages over 10 seeds:

| Method | Test Success | Avg Steps | Test Score | Nonzero Success Seeds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.3375 +/- 0.0800` | `22.4000 +/- 0.6799` | `0.4411 +/- 0.0393` | `10/10` |
| `genetic_q` | `0.1000 +/- 0.1458` | `95.2000 +/- 59.7675` | `0.2338 +/- 0.1468` | `3/10` |
| `q_learning` | `0.1125 +/- 0.1038` | `71.0000 +/- 59.6070` | `0.2247 +/- 0.1730` | `5/10` |
| `novelty_q` | `0.0125 +/- 0.0375` | `131.8000 +/- 36.6000` | `0.0816 +/- 0.0932` | `1/10` |

Per-run best by `test_score`:

```text
cyclic_novelty: 8/10 clear wins
genetic_q: 2/10 best by tie-breaking, effectively tied with cyclic_novelty
q_learning: 0/10
novelty_q: 0/10
```

CSV summary:

```text
outputs/ten_seed_summary.csv
```

## Hard 16x16 Results

Final averages over 10 seeds:

| Method | Test Success | Avg Steps | Test Score | Nonzero Success Seeds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.0813 +/- 0.0628` | `75.6333 +/- 90.1853` | `0.3010 +/- 0.0827` | `8/10` |
| `genetic_q` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.1340 +/- 0.0080` | `0/10` |
| `novelty_q` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0710 +/- 0.0416` | `0/10` |
| `q_learning` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0530 +/- 0.0400` | `0/10` |

Per-run best by `test_score`:

```text
cyclic_novelty: 8/10 clear wins
genetic_q: 1/10 clear win
cyclic_novelty and genetic_q: 1/10 tied at 0.15
q_learning: 0/10
novelty_q: 0/10
```

CSV summary:

```text
outputs/hard_summary.csv
```

## Hard Ablation: Replay and Restart

After the hard benchmark, four cyclic variants were tested:

| Variant | Replay Bank | Reproducibility Restart |
| --- | ---: | ---: |
| `cyclic_novelty` | no | no |
| `cyclic_replay` | yes | no |
| `cyclic_restart` | no | yes |
| `cyclic_replay_restart` | yes | yes |

The replay bank stores successful trajectories and reinforces their
state-action pairs during exploitation. Restart reconstructs the population from
hall-of-fame elites, replay-seeded policies, strongly mutated elites, and random
new policies when exploitation fails to reach a minimum success threshold.

The method RNG was changed to use fixed per-method offsets so subset runs are
not affected by method order.

Final hard ablation averages over 10 seeds:

| Method | Test Success | Avg Steps | Test Score | Nonzero Success Seeds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_replay` | `0.0938 +/- 0.0576` | `75.4667 +/- 90.2686` | `0.3141 +/- 0.0854` | `8/10` |
| `cyclic_novelty` | `0.0813 +/- 0.0628` | `98.1000 +/- 103.3716` | `0.2817 +/- 0.0903` | `7/10` |
| `cyclic_replay_restart` | `0.0688 +/- 0.0710` | `98.0500 +/- 103.4042` | `0.2779 +/- 0.0927` | `7/10` |
| `cyclic_restart` | `0.0625 +/- 0.0839` | `143.4000 +/- 112.6021` | `0.2403 +/- 0.1071` | `5/10` |

Per-run best by `test_score`:

```text
cyclic_replay: 5/10 wins
cyclic_novelty: 2/10 wins
cyclic_replay_restart: 2/10 wins
cyclic_restart: 1/10 wins
```

Current interpretation:

- Success replay is beneficial in the hard environment.
- Restart alone is not reliably beneficial in its current form.
- Replay plus restart does not yet outperform replay alone, suggesting the
  restart trigger or reconstruction mix is too disruptive.

CSV summary:

```text
outputs/ablation_hard_stable_summary.csv
```

## Interpretation

The current evidence supports a limited claim:

> In these tabular GridWorld experiments, cyclic novelty cultivation followed by
> exploitation and elite preservation outperforms simple Q-learning, novelty
> Q-learning, and a basic genetic Q-learning baseline.

The hard benchmark is especially informative because all baselines had zero
test success across 10 seeds, while `cyclic_novelty` solved at least one test
maze in 8/10 seeds.

The likely reason is that pure novelty expands behavior but does not preserve
useful solutions, while pure exploitation struggles to find sparse paths in
harder mazes. The cyclic method benefits from both:

- novelty creates diverse candidate trajectories,
- exploitation selects for success and speed,
- hall-of-fame prevents discovered solutions from being destroyed.

## Revised Cycle Interpretation

The MiniGrid pilot exposed an important trade-off:

- `cyclic_novelty` tended to produce faster successful paths.
- `cyclic_replay` tended to produce more nonzero successes, but with slower
  average paths.

The correct interpretation is not to make replay a speed-optimized RL replay
buffer. That would collapse the method back toward ordinary reinforcement
learning. Instead, replay should be treated as an independent cycle:

```text
novelty cycle -> replay cycle -> exploitation cycle
```

Each cycle has a separate role:

### Novelty Cycle

Purpose:

```text
Generate diverse candidate behavior.
```

It should emphasize:

- novelty,
- trajectory diversity,
- broad exploration,
- weak success signal.

It should not aggressively optimize speed.

### Replay Cycle

Purpose:

```text
Test whether discovered successes can be reproduced.
```

It should emphasize:

- repeated success,
- remembering successful state-action traces,
- robustness of discovered behaviors.

It should not directly optimize for the shortest path. Otherwise replay and
exploitation become the same phase.

### Exploitation Cycle

Purpose:

```text
Compress reproducible successful behavior into faster policies.
```

It should emphasize:

- success rate,
- speed,
- stability,
- elite preservation.

This phase is where "fast and correct" should dominate.

### Gate / Restart Logic

Restart should not be a blunt reset. Current ablation suggests that restart
alone is too disruptive. Future restart logic should be gated by replay failure:

```text
If novelty finds candidates but replay cannot reproduce success:
    continue replay or return to novelty.

If replay reproduces success but exploitation cannot improve speed:
    continue exploitation.

If all phases stall:
    restart part of the population while preserving replay-confirmed elites.
```

## Current Limitations

1. This is still a toy environment.
2. Policies are tabular Q-tables, not neural policies.
3. Baselines are simple and not tuned aggressively.
4. `stability` needs refinement, because stable failure can be rewarded.
5. The local observation design may bias which methods perform well.
6. Success rates on hard mazes are still low in absolute terms.
7. No statistical significance test has been run yet.
8. No ablation study has isolated the individual contribution of:
   - cyclic phase switching,
   - novelty bonus,
   - time penalty,
   - hall-of-fame,
   - exploitation score.

## Suggested Next Experiments

1. Ablation study:
   - no hall-of-fame,
   - no novelty phase,
   - no time penalty,
   - no success-gated exploitation score.
2. Improve restart:
   - trigger only after repeated validation failure,
   - preserve more replay-seeded elites,
   - reduce random population injection when replay exists.
3. Stronger baselines:
   - more episodes for Q-learning,
   - tuned genetic Q-learning,
   - MiniGrid baselines.
4. Larger held-out test set:
   - 50 or 100 test mazes per seed.
5. Alternative maze difficulty:
   - obstacle rates from `0.20` to `0.35`,
   - sizes `12`, `16`, `20`.
6. Replace tabular Q-learning with a small neural policy.

## MiniGrid Pilot

A first external benchmark adapter was added for MiniGrid. This is still a
tabular experiment: MiniGrid's 7x7x3 partial observation is compressed into a
small discrete feature vector using:

- agent direction,
- front/left/right object type,
- whether goal/key/door/lava is visible,
- whether the agent is carrying an object.

Only three MiniGrid actions are used in the first adapter:

```text
left, right, forward
```

The first pilot used `MiniGrid-FourRooms-v0`, 3 seeds, and a deliberately light
configuration:

```text
generations = 15
population = 8
episodes_per_agent = 2
eval_episodes = 6
```

Results:

| Method | Test Success | Avg Steps | Test Score | Nonzero Success Seeds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.1667 +/- 0.0000` | `15.0000 +/- 7.8740` | `0.3741 +/- 0.0092` | `3/3` |
| `cyclic_replay` | `0.2222 +/- 0.0785` | `66.5000 +/- 71.8830` | `0.3443 +/- 0.0416` | `3/3` |
| `genetic_q` | `0.1111 +/- 0.1571` | `172.6667 +/- 117.8511` | `0.1588 +/- 0.2245` | `1/3` |
| `q_learning` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0000 +/- 0.0000` | `0/3` |

Per-run best by `test_score`:

```text
cyclic_novelty: 2/3 wins
genetic_q: 1/3 wins
cyclic_replay: 0/3 wins by score, but 3/3 nonzero success
q_learning: 0/3
```

Interpretation:

- The custom GridWorld signal did not immediately disappear on MiniGrid.
- The current MiniGrid adapter is very rough and computationally slow.
- The result should be treated as a pilot, not as a definitive benchmark.
- A more convincing MiniGrid benchmark needs stronger feature extraction or a
  small neural policy.

CSV summary:

```text
outputs/minigrid_fourrooms_pilot_summary.csv
```

## MiniGrid Strong-Baseline Pilot

To avoid using overly weak traditional baselines, a stronger MiniGrid benchmark
was run with:

- `q_learning_strong`: 4x training episodes per generation.
- `genetic_q_strong`: 2x training episodes per individual plus stronger
  evaluation and lower mutation.
- `cyclic_three_phase`: explicit novelty -> replay -> exploitation cycles.

Task:

```text
MiniGrid-FourRooms-v0
```

Configuration:

```text
seeds = 7, 17, 27
generations = 20
population = 8
episodes_per_agent = 2
eval_episodes = 6
```

Results:

| Method | Test Success | Avg Steps | Test Score | Nonzero Success Seeds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.1667 +/- 0.0000` | `12.6667 +/- 6.2361` | `0.3768 +/- 0.0073` | `3/3` |
| `cyclic_three_phase` | `0.1667 +/- 0.0000` | `17.6667 +/- 8.3799` | `0.3710 +/- 0.0098` | `3/3` |
| `cyclic_replay` | `0.1667 +/- 0.0000` | `22.6667 +/- 21.4838` | `0.3651 +/- 0.0251` | `3/3` |
| `genetic_q_strong` | `0.1111 +/- 0.1571` | `173.3333 +/- 116.9083` | `0.1580 +/- 0.2234` | `1/3` |
| `q_learning_strong` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0000 +/- 0.0000` | `0/3` |

Per-run best by `test_score`:

```text
cyclic_novelty: 1/3 wins
cyclic_replay: 1/3 wins
genetic_q_strong: 1/3 wins
cyclic_three_phase: 0/3 wins
q_learning_strong: 0/3 wins
```

Interpretation:

- The cyclic methods still survive after strengthening the traditional
  baselines.
- `cyclic_three_phase` is viable but not yet better than the simpler cyclic
  variants.
- `cyclic_novelty` remains the fastest of the cyclic methods in this pilot.
- More MiniGrid work should focus on better observation features and a larger
  but resumable benchmark run.

CSV summary:

```text
outputs/minigrid_fourrooms_strong_summary.csv
```

## Explore-Operate-Replay Branch

The next active branch deprecates the replay-heavy direction and tests:

```text
explore -> operate -> explore -> operate -> replay
```

The first MiniGrid comparison includes stronger tabular baselines plus
lightweight Go-Explore and MAP-Elites style competitors.

Result summary over 3 seeds:

| Method | Test Success | Avg Steps | Test Score | Nonzero Success Seeds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.1667 +/- 0.0000` | `10.0000 +/- 2.9439` | `0.3799 +/- 0.0034` | `3/3` |
| `cyclic_novelty` | `0.1667 +/- 0.0000` | `12.6667 +/- 6.2361` | `0.3768 +/- 0.0073` | `3/3` |
| `map_elites_lite` | `0.1667 +/- 0.0000` | `26.3333 +/- 7.1336` | `0.3608 +/- 0.0083` | `3/3` |
| `genetic_q_strong` | `0.1111 +/- 0.1571` | `173.3333 +/- 116.9083` | `0.1580 +/- 0.2234` | `1/3` |
| `go_explore_lite` | `0.0556 +/- 0.0786` | `172.6667 +/- 117.8511` | `0.1282 +/- 0.1813` | `1/3` |
| `q_learning_strong` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0000 +/- 0.0000` | `0/3` |

Interpretation:

- `cyclic_operate_replay` slightly outperformed `cyclic_novelty` by speed.
- `map_elites_lite` is competitive on reliability but slower.
- `go_explore_lite` is currently too weak without a stronger robustification
  phase.
- The result supports continuing the delayed replay schedule, but does not yet
  prove superiority.

Runtime profiling was added after this result. New MiniGrid runs include:

```text
runtime_seconds
```

This should be used with `test_score` when judging future rolling-training
variants.

## MiniGrid Three-Minute Time-Limited Run

A follow-up run compared only:

```text
cyclic_operate_replay
cyclic_novelty
map_elites_lite
go_explore_lite
```

Each method received a 180-second wall-clock budget per seed.

Results over 3 seeds:

| Method | Test Success | Avg Steps | Test Score | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.2222 +/- 0.0785` | `15.6667 +/- 11.6142` | `0.4039 +/- 0.0524` | `182.0962 +/- 0.8830` |
| `map_elites_lite` | `0.2222 +/- 0.0785` | `19.5000 +/- 15.6897` | `0.3994 +/- 0.0251` | `181.7897 +/- 1.1568` |
| `cyclic_operate_replay` | `0.2222 +/- 0.0785` | `39.6667 +/- 45.5070` | `0.3757 +/- 0.0844` | `181.2513 +/- 0.2591` |
| `go_explore_lite` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0000 +/- 0.0000` | `53.2007 +/- 1.2788` |

This changes the interpretation: delayed replay was better in the fixed-count
run, but `cyclic_novelty` and `map_elites_lite` were stronger under equal
wall-clock budget.

## MiniGrid State Encoder Comparison

After extracting MiniGrid state encoders, compact and geometry state
representations were compared under the same 3-minute-per-method budget.

Compact state:

| Method | Test Success | Avg Steps | Test Score |
| --- | ---: | ---: | ---: |
| `cyclic_novelty` | `0.2222 +/- 0.0785` | `22.6667 +/- 21.4838` | `0.3957 +/- 0.0608` |
| `map_elites_lite` | `0.1667 +/- 0.0000` | `11.6667 +/- 4.9216` | `0.3780 +/- 0.0057` |
| `cyclic_operate_replay` | `0.2222 +/- 0.0785` | `39.6667 +/- 45.5070` | `0.3757 +/- 0.0844` |

Geometry state:

| Method | Test Success | Avg Steps | Test Score |
| --- | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `8.3333 +/- 0.9428` | `0.4430 +/- 0.0438` |
| `cyclic_novelty` | `0.2778 +/- 0.0785` | `13.0000 +/- 6.5701` | `0.4375 +/- 0.0507` |
| `map_elites_lite` | `0.3333 +/- 0.1361` | `61.6111 +/- 27.0303` | `0.4111 +/- 0.0463` |

Interpretation:

- Geometry state representation improved every tested method.
- `cyclic_operate_replay` improved the most and became the best average method
  under the fixed-time budget.
- This suggests the previous compact state was a real bottleneck for delayed
  replay.

## Ecological Cycle Pilot

An ecological cycle variant was tested:

```text
radiation -> niche -> stress -> bottleneck -> reradiation -> consolidation
```

It adds:

- niche archive,
- stress evaluation,
- bottleneck reconstruction,
- re-radiation from survivors,
- motif consolidation.

Result under geometry state and a 3-minute budget:

| Method | Test Success | Avg Steps | Test Score |
| --- | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `8.3333 +/- 0.9428` | `0.4430 +/- 0.0438` |
| `cyclic_ecology` | `0.2778 +/- 0.0785` | `9.0000 +/- 1.6330` | `0.4422 +/- 0.0432` |
| `cyclic_novelty` | `0.2778 +/- 0.0785` | `14.0000 +/- 5.6716` | `0.4364 +/- 0.0498` |
| `map_elites_lite` | `0.3333 +/- 0.1361` | `53.9444 +/- 25.5061` | `0.4201 +/- 0.0451` |

Interpretation:

- The ecological cycle is viable immediately, which is a positive signal.
- It does not yet clearly beat `cyclic_operate_replay`.
- The added complexity needs ablation before it should become the main branch.

## Ecological Cycle Ablation

A one-seed ablation was run on seed 97 using geometry state and the same
3-minute budget.

| Method | Test Success | Avg Steps | Test Score |
| --- | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.3333` | `7.5000` | `0.4745` |
| `cyclic_ecology_niche_only` | `0.3333` | `78.5000` | `0.3913` |
| `cyclic_ecology_no_bottleneck` | `0.1667` | `2.0000` | `0.3893` |
| `cyclic_ecology_no_motif` | `0.1667` | `2.0000` | `0.3893` |
| `cyclic_ecology_no_stress` | `0.1667` | `2.0000` | `0.3893` |

Interpretation:

- `cyclic_operate_replay` clearly wins this ablation seed.
- Niche-only exploration keeps success but is slow.
- Removing individual ecological mechanisms does not reveal a single obviously
  dominant component.
- Ecology should be treated as a promising but currently overcomplicated branch.

## Ecological Replay Variants

The full ecology design was split into three lighter variants and compared
against the current `cyclic_operate_replay` baseline.

Setup:

```text
MiniGrid-FourRooms-v0
state_encoder = geometry
seeds = 7, 17, 27
method_time_limit_seconds = 180
```

Results:

| Method | Test Success | Avg Steps | Test Score |
| --- | ---: | ---: | ---: |
| `cyclic_motif_oriented_radiation` | `0.3333 +/- 0.0000` | `19.5000 +/- 6.1779` | `0.4605 +/- 0.0072` |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `8.3333 +/- 0.9428` | `0.4430 +/- 0.0438` |
| `cyclic_speciated_stress_replay` | `0.2778 +/- 0.0785` | `8.8333 +/- 2.3921` | `0.4424 +/- 0.0443` |
| `cyclic_resource_ecology_replay` | `0.2778 +/- 0.0785` | `10.0000 +/- 2.5495` | `0.4410 +/- 0.0425` |

Per-run winners:

```text
seed 7:  cyclic_speciated_stress_replay
seed 17: cyclic_operate_replay
seed 27: cyclic_motif_oriented_radiation
```

Interpretation:

- `cyclic_motif_oriented_radiation` produced the best average score and the only
  flat `3/3` success rate in this comparison.
- `cyclic_operate_replay` remains the clean speed baseline.
- `cyclic_speciated_stress_replay` showed the clearest stress-gate signal but did
  not yet beat the baseline on average.
- `cyclic_resource_ecology_replay` maintained many active niches, but that
  diversity did not convert into a score advantage.
- The next serious direction is motif-guided radiation plus stricter speed
  pressure, not the full ecological loop.

## Reproduction Commands

Setup:

```bash
uv sync
```

Environment check:

```bash
uv run python scripts/check_env.py
```

Run one default experiment:

```bash
uv run python src/train.py --config configs/default.toml --seed 7 --output-dir outputs/example_default
```

Run one hard experiment:

```bash
uv run python src/train.py --config configs/hard.toml --seed 7 --output-dir outputs/example_hard
```

Summarize multiple runs:

```bash
uv run python scripts/summarize_runs.py outputs/hard_seed_7 outputs/hard_seed_17 --csv-out outputs/example_summary.csv
```
