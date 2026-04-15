# Versioning Plan

Use git to preserve code, configuration, and written experiment conclusions.
Generated run artifacts stay local under `outputs/` unless a specific artifact is
promoted into a tracked document.

## What To Commit

- Source code in `src/`
- Scripts in `scripts/`
- Experiment configs in `configs/`
- Documentation in `docs/`
- Dependency files:
  - `pyproject.toml`
  - `uv.lock`
  - `uv.toml`
  - `requirements.txt`

## What Not To Commit

- `.venv/`
- `.uv-cache/`
- `.mplconfig/`
- `.cache/`
- Generated files under `outputs/`

## Suggested Branches

- `main`: stable checkpoints.
- `experiment/replay-bank`: success replay changes.
- `experiment/restart-policy`: restart trigger and population reconstruction.
- `experiment/minigrid`: future MiniGrid environment migration.
- `experiment/nn-policy`: future neural policy experiments.

## Suggested Checkpoints

Create a commit after each coherent experimental state:

1. Environment and baseline scaffold.
2. Cyclic novelty with hall-of-fame.
3. Hard benchmark report.
4. Replay/restart ablation.
5. Any future algorithmic change before rerunning large experiments.

## Before A Run

```bash
uv run ruff check src scripts
uv run pyright src scripts
uv run python scripts/check_env.py
```

## After A Run

1. Save generated artifacts in a unique output directory.
2. Summarize with `scripts/summarize_runs.py`.
3. Copy stable conclusions into `docs/experiment_report.md`.
4. Commit source/config/doc changes.
