# Maze Novelty

This is a small experimental playground for staged novelty exploration in a
12x12 GridWorld.

The first version compares four methods:

- `q_learning`: ordinary Q-learning.
- `novelty_q`: Q-learning with a novelty bonus.
- `genetic_q`: population-based Q-learning with mutation/crossover.
- `cyclic_novelty`: cyclic novelty cultivation followed by exploitation.

Setup:

```bash
uv sync
```

Environment check without training:

```bash
uv run python src/train.py --config configs/quick.toml --dry-run
uv run python scripts/check_env.py
```

Quick smoke run:

```bash
uv run python src/train.py --config configs/quick.toml --no-plots
```

Plot smoke run:

```bash
uv run python src/train.py --config configs/plot_smoke.toml
```

Full default run:

```bash
uv run python src/train.py --config configs/default.toml
```

Hard run:

```bash
uv run python src/train.py --config configs/hard.toml
```

MiniGrid FourRooms pilot:

```bash
uv run python scripts/run_minigrid_benchmark.py --env-id MiniGrid-FourRooms-v0 --name fourrooms_pilot
```

MiniGrid benchmark plan:

- `docs/minigrid_benchmark_plan.md`
- `docs/explore_operate_benchmark_plan.md`

Frozen result checkpoints:

- `docs/v0.3_frozen_result.md`

Outputs are written to `outputs/`:

- `metrics.csv`
- `summary.png`
- `paths.png`

Current experiment report:

- `docs/experiment_report.md`
- `docs/version_history.md`
- `docs/pretraining_transfer_framework.md`

Versioning notes:

- `docs/versioning.md`
- `docs/framework_upgrade_plan.md`

Quality checks:

```bash
uv run ruff check src scripts tests
uv run pyright src scripts
uv run pytest
```

The experiment is intentionally small and dependency-light. It is meant to test
whether the training loop produces useful behavior before moving to larger
environments.

Notes:

- `configs/quick.toml` is for checking that the environment and imports work.
- `configs/default.toml` is the first real experiment preset.
- Matplotlib and font caches are redirected into project-local cache folders to
  avoid writing into user-level cache directories.
