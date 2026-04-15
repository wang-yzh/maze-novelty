# Explore-Operate Benchmark Plan

This branch tests the next active cycle design:

```text
explore -> operate -> explore -> operate -> replay
```

## Rationale

The previous replay-heavy branch is deprecated. Replay should not appear in every
large cycle, because that makes the method drift toward ordinary reinforcement
learning and can slow down discovered successful paths.

The new hypothesis is:

```text
Two exploration/operation pairs should create and compress behavior before a
single replay phase tests reproducibility.
```

## Methods

Active methods:

```text
q_learning_strong
genetic_q_strong
cyclic_novelty
cyclic_operate_replay
go_explore_lite
map_elites_lite
```

Deprecated methods:

```text
cyclic_replay
cyclic_three_phase
```

They remain in code for historical reproduction, but should not be used as the
main comparison set.

## Lightweight Competitors

### `go_explore_lite`

This is a lightweight first-phase Go-Explore style baseline:

- keep an archive of reached cells,
- select an archived cell,
- return to it by replaying the stored action sequence,
- explore randomly from there,
- keep a better trajectory for a cell when found.

It does not implement full robustification. That makes it light enough for the current
machine, but also weaker than full Go-Explore. A minimal tabular
robustification step distills successful archived traces into a Q-table.

### `map_elites_lite`

This is a lightweight MAP-Elites / quality-diversity baseline:

- keep one elite Q-agent per behavior descriptor,
- describe behavior by final-position bins and speed bin,
- sample elites,
- mutate and train candidates,
- replace a cell only when candidate quality improves.

This avoids adding a heavier QD dependency while preserving the main idea of
quality-diversity search.

## Smoke Run

Use this before any large run:

```bash
uv run python scripts/run_minigrid_benchmark.py \
  --env-id MiniGrid-FourRooms-v0 \
  --name explore_operate_smoke \
  --seeds 7 \
  --generations 6 \
  --population 6 \
  --episodes-per-agent 1 \
  --eval-episodes 4 \
  --eval-every 3 \
  --methods q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_operate_replay,go_explore_lite,map_elites_lite
```

## First Real Run

This is the first comparison that should be interpreted.

```bash
uv run python scripts/run_minigrid_benchmark.py \
  --env-id MiniGrid-FourRooms-v0 \
  --name explore_operate_first \
  --seeds 7,17,27 \
  --generations 20 \
  --population 8 \
  --episodes-per-agent 2 \
  --eval-episodes 6 \
  --eval-every 5 \
  --methods q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_operate_replay,go_explore_lite,map_elites_lite
```

## Interpretation Rules

1. If `cyclic_operate_replay` beats `cyclic_novelty`, the delayed replay schedule
   is promising.
2. If `go_explore_lite` wins, archive-return exploration is a stronger competitor
   than our current novelty pressure.
3. If `map_elites_lite` wins, the QD framing may be the better research family.
4. If all methods are weak, improve MiniGrid observation features before scaling.
5. Compare `runtime_seconds` together with score. A method that only wins by
   spending far more wall-clock time is less useful for rolling training.

## Runtime Profiling

MiniGrid runs now record method-level wall-clock time:

```text
runtime_seconds
```

The field is written into each method row in `metrics.csv` and summarized by
`scripts/summarize_runs.py`.

Runtime smoke command:

```bash
uv run python scripts/run_minigrid_benchmark.py \
  --env-id MiniGrid-FourRooms-v0 \
  --name runtime_smoke \
  --seeds 7 \
  --generations 3 \
  --population 4 \
  --episodes-per-agent 1 \
  --eval-episodes 3 \
  --eval-every 3 \
  --methods q_learning_strong,genetic_q_strong,cyclic_novelty,cyclic_operate_replay,go_explore_lite,map_elites_lite
```

Smoke runtime result:

| Method | Runtime Seconds |
| --- | ---: |
| `genetic_q_strong` | `7.1351` |
| `cyclic_operate_replay` | `4.0111` |
| `cyclic_novelty` | `3.8735` |
| `map_elites_lite` | `3.4240` |
| `q_learning_strong` | `1.8786` |
| `go_explore_lite` | `0.7342` |

This smoke run is too small to interpret model quality. It is only a check that
runtime is captured correctly and a first signal that `genetic_q_strong` is the
largest runtime cost.

## Three-Minute Time-Limited Run

This run compares only the active internal methods and lightweight external
competitors under a wall-clock budget:

```text
3 minutes per method per seed
```

Command:

```bash
uv run python scripts/run_minigrid_benchmark.py \
  --env-id MiniGrid-FourRooms-v0 \
  --name time_limit_3min \
  --seeds 7,17,27 \
  --generations 100 \
  --population 8 \
  --episodes-per-agent 2 \
  --eval-episodes 6 \
  --eval-every 5 \
  --method-time-limit-seconds 180 \
  --methods cyclic_operate_replay,cyclic_novelty,map_elites_lite,go_explore_lite
```

The summarizer was updated for time-limited runs. It now selects the final row
per `(run, method)`, because different methods can stop at different generations.

Results:

| Method | Test Success | Avg Steps | Test Score | Runtime Seconds | Time-Limited Seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cyclic_novelty` | `0.2222 +/- 0.0785` | `15.6667 +/- 11.6142` | `0.4039 +/- 0.0524` | `182.0962 +/- 0.8830` | `3/3` |
| `map_elites_lite` | `0.2222 +/- 0.0785` | `19.5000 +/- 15.6897` | `0.3994 +/- 0.0251` | `181.7897 +/- 1.1568` | `3/3` |
| `cyclic_operate_replay` | `0.2222 +/- 0.0785` | `39.6667 +/- 45.5070` | `0.3757 +/- 0.0844` | `181.2513 +/- 0.2591` | `3/3` |
| `go_explore_lite` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0000 +/- 0.0000` | `53.2007 +/- 1.2788` | `0/3` |

Per-run best:

```text
seed 7:  cyclic_novelty
seed 17: map_elites_lite
seed 27: map_elites_lite
```

Final generation reached:

| Seed | `cyclic_novelty` | `cyclic_operate_replay` | `map_elites_lite` | `go_explore_lite` |
| --- | ---: | ---: | ---: | ---: |
| `7` | `55` | `45` | `80` | `100` |
| `17` | `50` | `45` | `80` | `100` |
| `27` | `45` | `45` | `65` | `100` |

Interpretation:

- Under fixed generation count, `cyclic_operate_replay` slightly led.
- Under fixed wall-clock budget, `cyclic_novelty` led, with `map_elites_lite`
  very close.
- `cyclic_operate_replay` was dragged down by seed 17, where it found a much
  slower successful path.
- `map_elites_lite` is stronger under time budget than the earlier fixed-count
  comparison suggested.
- `go_explore_lite` is fast but ineffective in this setting.

## First Real Run Result

The first real run completed on:

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
| `cyclic_operate_replay` | `0.1667 +/- 0.0000` | `10.0000 +/- 2.9439` | `0.3799 +/- 0.0034` | `3/3` |
| `cyclic_novelty` | `0.1667 +/- 0.0000` | `12.6667 +/- 6.2361` | `0.3768 +/- 0.0073` | `3/3` |
| `map_elites_lite` | `0.1667 +/- 0.0000` | `26.3333 +/- 7.1336` | `0.3608 +/- 0.0083` | `3/3` |
| `genetic_q_strong` | `0.1111 +/- 0.1571` | `173.3333 +/- 116.9083` | `0.1580 +/- 0.2234` | `1/3` |
| `go_explore_lite` | `0.0556 +/- 0.0786` | `172.6667 +/- 117.8511` | `0.1282 +/- 0.1813` | `1/3` |
| `q_learning_strong` | `0.0000 +/- 0.0000` | `256.0000 +/- 0.0000` | `0.0000 +/- 0.0000` | `0/3` |

Per-run best:

```text
seed 7: genetic_q_strong
seed 17: cyclic_operate_replay
seed 27: cyclic_novelty
```

Interpretation:

- Delaying replay until after two explore/operate pairs did not hurt success.
- `cyclic_operate_replay` slightly improved speed over `cyclic_novelty` in this
  run.
- `map_elites_lite` is a serious lightweight competitor because it achieved
  nonzero success in all seeds, but it was slower.
- `go_explore_lite` needs a stronger return/robustification implementation before
  it is a serious competitor on stochastic evaluation seeds.
- The result is still small. Treat it as a direction signal, not proof.

CSV summary:

```text
outputs/minigrid_explore_operate_first_summary.csv
```

## External References

- Go-Explore: Ecoffet et al., "Go-Explore: a New Approach for Hard-Exploration
  Problems", arXiv:1901.10995.
- MAP-Elites: Mouret and Clune, "Illuminating search spaces by mapping elites",
  arXiv:1504.04909.
- pyribs is a mature lightweight Python library for quality-diversity
  optimization, but this branch uses an internal MAP-Elites baseline to avoid a
  new dependency.
