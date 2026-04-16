# Framework Upgrade Plan

This branch starts the architecture upgrade from hard-coded experiment scripts
toward a small experiment framework.

## Goal

Make it cheap to test:

```text
method x phase schedule x benchmark x budget
```

without duplicating large training functions.

## Current Problem

`src/minigrid_train.py` currently mixes:

- method registry,
- method runtime timing,
- time-budget logic,
- MiniGrid training loops,
- cyclic method implementations,
- Go-Explore lite,
- MAP-Elites lite,
- CSV metric output.

This is workable for a few experiments, but too rigid for broader search over
phase schedules and benchmarks.

## Step 1: Completed In This Branch

The first low-risk extraction adds:

```text
src/core/budget.py
src/core/recording.py
src/core/method.py
```

Responsibilities:

- `MethodBudget`: method-level wall-clock budget and expiry checks.
- `annotate_method_rows`: attach `runtime_seconds` and `time_limited` to rows.
- `MethodRunner`: minimal protocol for future method registry work.

The existing MiniGrid algorithms are intentionally left in place so benchmark
behavior remains comparable to `v0.4.2-time-limited-benchmark`.

## Step 2: State Encoder Extraction

MiniGrid state encoding is now selectable:

```text
--state-encoder compact
--state-encoder geometry
```

Encoders:

- `MiniGridCompactEncoder`: the historical feature set.
- `MiniGridGeometryEncoder`: local geometry, visible-goal direction, topology
  type, and last action.

This makes state representation an explicit experiment dimension instead of a
hard-coded part of the environment wrapper.

Smoke checks:

```bash
uv run python src/minigrid_train.py --env-id MiniGrid-FourRooms-v0 --seed 7 --output-dir outputs/encoder_compact_smoke --generations 3 --population 4 --episodes-per-agent 1 --eval-episodes 3 --eval-every 3 --state-encoder compact --methods cyclic_novelty

uv run python src/minigrid_train.py --env-id MiniGrid-FourRooms-v0 --seed 7 --output-dir outputs/encoder_geometry_smoke --generations 3 --population 4 --episodes-per-agent 1 --eval-episodes 3 --eval-every 3 --state-encoder geometry --methods cyclic_novelty
```

Both completed successfully.

## State Encoder Time-Limited Comparison

Comparison setup:

```text
MiniGrid-FourRooms-v0
seeds = 7, 17, 27
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 180
methods = cyclic_novelty, cyclic_operate_replay, map_elites_lite
```

Compact encoder results:

| Method | Test Success | Avg Steps | Test Score | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.2222 +/- 0.0785` | `22.6667 +/- 21.4838` | `0.3957 +/- 0.0608` | `182.0284 +/- 1.2429` |
| `map_elites_lite` | `0.1667 +/- 0.0000` | `11.6667 +/- 4.9216` | `0.3780 +/- 0.0057` | `182.8052 +/- 1.8750` |
| `cyclic_operate_replay` | `0.2222 +/- 0.0785` | `39.6667 +/- 45.5070` | `0.3757 +/- 0.0844` | `181.6400 +/- 1.4099` |

Geometry encoder results:

| Method | Test Success | Avg Steps | Test Score | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `8.3333 +/- 0.9428` | `0.4430 +/- 0.0438` | `181.1481 +/- 0.5796` |
| `cyclic_novelty` | `0.2778 +/- 0.0785` | `13.0000 +/- 6.5701` | `0.4375 +/- 0.0507` | `181.1778 +/- 0.4350` |
| `map_elites_lite` | `0.3333 +/- 0.1361` | `61.6111 +/- 27.0303` | `0.4111 +/- 0.0463` | `180.8006 +/- 0.5870` |

Interpretation:

- Geometry state improves all three methods under a fixed time budget.
- `cyclic_operate_replay` benefits the most, moving from `0.3757` to `0.4430`.
- Under compact state, fixed-time comparison favored `cyclic_novelty`.
- Under geometry state, `cyclic_operate_replay` becomes the best average method.
- `map_elites_lite` reaches the highest average success rate under geometry, but
  its successful paths are slower.

This strongly suggests the delayed replay method was being constrained by the
older compact state representation.

## Next Steps

1. Extract a real `Method` class interface.
2. Move Q-learning, cyclic, Go-Explore lite, and MAP-Elites lite into
   `src/methods/`.
3. Extract phase implementations:

```text
explore
operate
replay
exploit
```

4. Make cyclic schedules config-driven:

```text
["explore", "operate", "explore", "operate", "replay"]
```

5. Add environment adapters under `src/envs/`.

## Design Constraint

Every refactor step must preserve a runnable benchmark. Do not break historical
tags; add new runners or compatibility wrappers where needed.

## Ecological Cycle Variant

An experimental method was added:

```text
cyclic_ecology
```

It tests an ecological pressure schedule:

```text
radiation -> niche -> stress -> bottleneck -> reradiation -> consolidation
```

Implemented mechanisms:

- `NicheArchive`: stores elites by behavior descriptor.
- `MotifBank`: extracts short successful trajectory segments.
- stress evaluation: reduced step budget and shifted evaluation seeds.
- bottleneck reconstruction: stress survivors, niche elites, mutated champions,
  and a small amount of fresh population.
- consolidation: success replay plus motif reinforcement.

Smoke command:

```bash
uv run python src/minigrid_train.py --env-id MiniGrid-FourRooms-v0 --seed 7 --output-dir outputs/ecology_smoke --generations 12 --population 6 --episodes-per-agent 1 --eval-episodes 3 --eval-every 3 --state-encoder geometry --method-time-limit-seconds 20 --methods cyclic_ecology
```

The smoke run completed successfully.

## Ecological Cycle Benchmark

Comparison setup:

```text
MiniGrid-FourRooms-v0
state_encoder = geometry
seeds = 7, 17, 27
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 180
methods = cyclic_novelty, cyclic_operate_replay, map_elites_lite, cyclic_ecology
```

Results:

| Method | Test Success | Avg Steps | Test Score | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `8.3333 +/- 0.9428` | `0.4430 +/- 0.0438` | `180.6715 +/- 0.5636` |
| `cyclic_ecology` | `0.2778 +/- 0.0785` | `9.0000 +/- 1.6330` | `0.4422 +/- 0.0432` | `181.5018 +/- 1.3472` |
| `cyclic_novelty` | `0.2778 +/- 0.0785` | `14.0000 +/- 5.6716` | `0.4364 +/- 0.0498` | `180.8439 +/- 0.4611` |
| `map_elites_lite` | `0.3333 +/- 0.1361` | `53.9444 +/- 25.5061` | `0.4201 +/- 0.0451` | `182.5798 +/- 2.7146` |

Per-run best:

```text
seed 7:  cyclic_ecology
seed 17: cyclic_operate_replay
seed 27: cyclic_ecology and cyclic_operate_replay tied by score
```

Interpretation:

- `cyclic_ecology` is viable on the first implementation.
- It nearly matches the current best `cyclic_operate_replay`, but does not clearly
  beat it on average.
- Its average successful path length is stable and short.
- The extra ecological phases are not yet obviously worth their complexity.
- The next useful step is ablation, especially:
  - remove motif consolidation,
  - remove stress,
  - keep niche archive only,
  - compare bottleneck ratios.

## Ecological Cycle Ablation: Seed 97

Purpose:

```text
Use one larger seed to compare ecological ablations against the current
cyclic_operate_replay baseline.
```

Setup:

```text
MiniGrid-FourRooms-v0
state_encoder = geometry
seed = 97
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 180
```

Methods:

```text
cyclic_operate_replay
cyclic_ecology_no_motif
cyclic_ecology_no_stress
cyclic_ecology_niche_only
cyclic_ecology_no_bottleneck
```

Results:

| Method | Test Success | Avg Steps | Test Score | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_operate_replay` | `0.3333` | `7.5000` | `0.4745` | `180.5210` |
| `cyclic_ecology_niche_only` | `0.3333` | `78.5000` | `0.3913` | `158.5618` |
| `cyclic_ecology_no_bottleneck` | `0.1667` | `2.0000` | `0.3893` | `181.1038` |
| `cyclic_ecology_no_motif` | `0.1667` | `2.0000` | `0.3893` | `181.0492` |
| `cyclic_ecology_no_stress` | `0.1667` | `2.0000` | `0.3893` | `181.1940` |

Interpretation:

- The current `cyclic_operate_replay` baseline clearly wins this seed.
- `niche_only` preserves success rate but produces much slower successful paths.
- Removing motif, stress, or bottleneck each produced a very fast successful path,
  but at lower success rate.
- No single ablation explains the full ecological cycle's earlier near-tie with
  `cyclic_operate_replay`.
- The ecological method likely needs schedule/ratio tuning rather than simply
  removing one component.

## V0.7 Ecological Replay Variants

The full ecological loop was split into three lighter variants on top of the
`cyclic_operate_replay` baseline:

```text
cyclic_speciated_stress_replay
cyclic_motif_oriented_radiation
cyclic_resource_ecology_replay
```

Shared lower-level modules:

- richer behavior descriptors: final region, success, speed, coverage, turn
  pattern, revisit ratio, and progress ratio.
- `NicheArchive`: per-descriptor elite storage with local survivor selection.
- `MotifBank`: successful trajectory segment storage and reinforcement.
- stress scoring and internal metrics: `active_niches`, `stress_pass_rate`,
  `replay_bank_size`, and `motif_count`.

Variant intent:

- `cyclic_speciated_stress_replay`: add niche sorting and a stress gate before
  replay.
- `cyclic_motif_oriented_radiation`: reduce blind exploration by anchoring
  radiation around successful motifs.
- `cyclic_resource_ecology_replay`: add niche crowding pressure and resource
  competition.

Benchmark setup:

```text
MiniGrid-FourRooms-v0
state_encoder = geometry
seeds = 7, 17, 27
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 180
```

Results:

| Method | Test Success | Avg Steps | Test Score | Active Niches | Stress Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cyclic_motif_oriented_radiation` | `0.3333 +/- 0.0000` | `19.5000 +/- 6.1779` | `0.4605 +/- 0.0072` | `144.0000 +/- 14.6969` | `0.0000 +/- 0.0000` |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `8.3333 +/- 0.9428` | `0.4430 +/- 0.0438` | `0.0000 +/- 0.0000` | `0.0000 +/- 0.0000` |
| `cyclic_speciated_stress_replay` | `0.2778 +/- 0.0785` | `8.8333 +/- 2.3921` | `0.4424 +/- 0.0443` | `100.6667 +/- 2.8674` | `0.3333 +/- 0.4714` |
| `cyclic_resource_ecology_replay` | `0.2778 +/- 0.0785` | `10.0000 +/- 2.5495` | `0.4410 +/- 0.0425` | `112.6667 +/- 3.6818` | `0.0000 +/- 0.0000` |

Per-run best:

```text
seed 7:  cyclic_speciated_stress_replay
seed 17: cyclic_operate_replay
seed 27: cyclic_motif_oriented_radiation
```

Interpretation:

- `cyclic_motif_oriented_radiation` is the first v0.7 variant to beat
  `cyclic_operate_replay` on average score in the 3-seed geometry benchmark.
- `cyclic_operate_replay` remains the fastest average successful path.
- `cyclic_speciated_stress_replay` reached the best score on seed 7 and is the
  only variant with nonzero average stress pass rate.
- `cyclic_resource_ecology_replay` preserved diversity but did not improve final
  score over the baseline.
- The result supports keeping `operate-replay` as the engineering baseline while
  treating motif-guided radiation as the strongest new branch.

## V0.8 Speed-Disciplined Motif Replay

The v0.8 branch tests whether motif-guided radiation can inherit the speed
discipline of `cyclic_operate_replay`.

New method:

```text
cyclic_motif_fast_replay
```

Core changes relative to `cyclic_motif_oriented_radiation`:

- same high-level cycle:
  `anchored_explore -> operate -> free_explore -> operate -> fast_replay`.
- only short successful rollouts enter the fast replay and motif banks.
- operate and replay selection use higher speed pressure.
- new diagnostic metrics:
  `fast_success_count`, `fast_replay_ratio`,
  `first_success_generation`, and `best_score_generation`.

Benchmark setup:

```text
MiniGrid-FourRooms-v0
state_encoder = geometry
seeds = 7, 17, 27
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 100
methods = cyclic_operate_replay,
          cyclic_motif_oriented_radiation,
          cyclic_motif_fast_replay,
          cyclic_speciated_stress_replay
```

Results:

| Method | Test Success | Avg Steps | Test Score | Fast Replay Ratio |
| --- | ---: | ---: | ---: | ---: |
| `cyclic_motif_fast_replay` | `0.3333 +/- 0.0000` | `18.1667 +/- 7.0985` | `0.4621 +/- 0.0083` | `0.2466 +/- 0.0259` |
| `cyclic_speciated_stress_replay` | `0.2778 +/- 0.0785` | `18.5000 +/- 14.4280` | `0.4311 +/- 0.0345` | `0.0000 +/- 0.0000` |
| `cyclic_motif_oriented_radiation` | `0.2778 +/- 0.0785` | `22.1667 +/- 8.0243` | `0.4268 +/- 0.0340` | `0.0000 +/- 0.0000` |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `24.5000 +/- 12.0899` | `0.4241 +/- 0.0310` | `0.0000 +/- 0.0000` |

Per-run best:

```text
seed 7:  cyclic_motif_fast_replay
seed 17: cyclic_motif_fast_replay
seed 27: cyclic_motif_fast_replay
```

Interpretation:

- `cyclic_motif_fast_replay` won all three seeds under the shorter 100-second
  budget.
- It kept the flat `3/3` success profile from motif-guided radiation while
  improving speed.
- The result supports the v0.8 hypothesis: the useful direction is not adding
  more phases, but adding speed discipline to motif-guided exploration.
- `cyclic_operate_replay` remains an important baseline, but under this budget it
  lost both average score and average path length to `cyclic_motif_fast_replay`.

## V0.9 Motif Memory Quality

The v0.9 branch tests whether `cyclic_motif_fast_replay` can be improved by
raising memory quality rather than changing the phase cycle.

New method:

```text
cyclic_motif_fast_replay_v2
```

Core changes:

- `ScoredMotifBank`: motifs are ranked by parent success, parent speed, segment
  mobility, novelty, and motif type.
- success is split into fast and medium tiers.
- high-progress and high-mobility failures can produce candidate motifs.
- anchored and free exploration use separate scoring functions.
- operate training adds small revisit and turn penalties.
- diagnostics include confirmed/candidate motif counts, motif fill generation,
  and replay improvement delta.

Benchmark setup:

```text
MiniGrid-FourRooms-v0
state_encoder = geometry
seeds = 7, 17, 27
generations = 100
population = 8
episodes_per_agent = 2
eval_episodes = 6
eval_every = 5
method_time_limit_seconds = 100
methods = cyclic_operate_replay,
          cyclic_motif_oriented_radiation,
          cyclic_motif_fast_replay,
          cyclic_motif_fast_replay_v2
```

Results:

| Method | Test Success | Avg Steps | Test Score | Fast Replay Ratio | Replay Improvement Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cyclic_motif_fast_replay` | `0.3333 +/- 0.0000` | `17.8333 +/- 7.4199` | `0.4624 +/- 0.0087` | `0.2448 +/- 0.0255` | `0.0000 +/- 0.0000` |
| `cyclic_motif_fast_replay_v2` | `0.2778 +/- 0.0785` | `20.3333 +/- 15.7551` | `0.4289 +/- 0.0333` | `0.3001 +/- 0.0389` | `-0.1282 +/- 0.0453` |
| `cyclic_motif_oriented_radiation` | `0.2778 +/- 0.0785` | `22.1667 +/- 8.0243` | `0.4268 +/- 0.0340` | `0.0000 +/- 0.0000` | `0.0000 +/- 0.0000` |
| `cyclic_operate_replay` | `0.2778 +/- 0.0785` | `24.5000 +/- 12.0899` | `0.4241 +/- 0.0310` | `0.0000 +/- 0.0000` | `0.0000 +/- 0.0000` |

Per-run best:

```text
seed 7:  cyclic_motif_fast_replay
seed 17: cyclic_motif_fast_replay
seed 27: cyclic_motif_fast_replay
```

Interpretation:

- `cyclic_motif_fast_replay_v2` did not beat the v0.8 champion.
- The v2 motif bank filled immediately and saturated at 180 motifs, but that did
  not translate into better final policy quality.
- The negative replay improvement delta suggests the scored/candidate motif
  replay is currently too noisy or too strong.
- The added medium-success and candidate-motif channels likely diluted the clean
  short-success signal that made v0.8 strong.
- The next step should not be more memory complexity. It should either simplify
  v2 back toward fast-only confirmed motifs or use v2 diagnostics to tune replay
  strength before rerunning.
