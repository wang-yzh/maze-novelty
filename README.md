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

Outputs are written to `outputs/`:

- `metrics.csv`
- `summary.png`
- `paths.png`

Current experiment report:

- `docs/experiment_report.md`

Versioning notes:

- `docs/versioning.md`

The experiment is intentionally small and dependency-light. It is meant to test
whether the training loop produces useful behavior before moving to larger
environments.

Notes:

- `configs/quick.toml` is for checking that the environment and imports work.
- `configs/default.toml` is the first real experiment preset.
- Matplotlib and font caches are redirected into project-local cache folders to
  avoid writing into user-level cache directories.
