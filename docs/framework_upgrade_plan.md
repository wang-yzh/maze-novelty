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
