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
