# MiniGrid Benchmark Plan

This plan is for a more credible benchmark after the initial FourRooms pilot.

## Goal

Test whether the cyclic methods remain competitive against stronger traditional
baselines on a standard MiniGrid task, without changing the custom GridWorld
baseline branch.

## Environment

Initial task:

```text
MiniGrid-FourRooms-v0
```

Why this task:

- standard MiniGrid environment,
- sparse reward,
- navigation/exploration pressure,
- lighter than DoorKey or LavaCrossing.

## Methods

Run these methods:

```text
q_learning_strong
genetic_q_strong
cyclic_novelty
cyclic_replay
cyclic_three_phase
```

Strong baselines:

- `q_learning_strong`: 4x the training episodes per generation versus the basic
  Q-learning baseline.
- `genetic_q_strong`: 2x training episodes per individual, stronger evaluation,
  larger elite fraction, lower mutation during refinement.

New cyclic method:

- `cyclic_three_phase`: separate novelty, replay, and exploitation cycles.

## Conservative Run

This is intended to finish on the current machine without excessive runtime.

```bash
uv run python scripts/run_minigrid_benchmark.py \
  --env-id MiniGrid-FourRooms-v0 \
  --name fourrooms_strong \
  --seeds 7,17,27 \
  --generations 20 \
  --population 8 \
  --episodes-per-agent 2 \
  --eval-episodes 6 \
  --eval-every 5 \
  --methods q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_replay,cyclic_three_phase
```

The runner is resumable. If interrupted, rerun the same command and completed
seed directories will be skipped.

## Stronger Run

Only run this after the conservative run is informative.

```bash
uv run python scripts/run_minigrid_benchmark.py \
  --env-id MiniGrid-FourRooms-v0 \
  --name fourrooms_strong_5seed \
  --seeds 7,17,27,37,47 \
  --generations 30 \
  --population 10 \
  --episodes-per-agent 3 \
  --eval-episodes 8 \
  --eval-every 5 \
  --methods q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_replay,cyclic_three_phase
```

## Interpretation Rules

Do not treat the MiniGrid result as final proof. Use it to decide:

1. Whether the cyclic methods survive a standard environment.
2. Whether the three-phase split improves the replay/speed trade-off.
3. Whether stronger baselines erase the current advantage.
4. Whether the tabular observation adapter is too weak and should be replaced.

## Expected Failure Modes

- All methods fail: feature extraction is too poor or budget is too small.
- Strong genetic baseline wins: cyclic method may need better replay/exploitation
  separation.
- `cyclic_replay` wins success but not speed: replay improves reproduction but
  still needs exploitation compression.
- `cyclic_three_phase` wins: supports the separate novelty/replay/exploitation
  cycle hypothesis.
