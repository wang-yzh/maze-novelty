# Project Closure 2026-04-20

## Status

This project is archived.

Final active branch:

```text
experiment/behavior-prior-transfer
```

Final local closure commit before this document:

```text
6d8bc86 Ignore shared virtualenv symlink
```

The project should not continue as the main research line. It remains useful as
a historical record and as a source of implementation pieces for the next
artificial-life project.

## Why It Stopped

The project began as cyclic novelty and evolutionary reinforcement learning for
maze solving. It later shifted into pretraining-transfer research:

```text
pretraining cycle
-> behavior artifact
-> target probe
-> behavior-prior extraction
-> adaptation evaluation
```

That shift was technically coherent, but it also moved away from the original
goal.

The original goal was closer to:

```text
raise a persistent population that inherits experience across generations
```

The project instead became:

```text
test whether a training artifact improves benchmark transfer
```

That is a valid machine-learning research question, but it is not the same as
artificial life.

## Final Research Answer

The final honest answer is:

```text
The system can create, record, and execute behavior structures.
It has not shown stable pretraining-transfer benefit.
It does not yet model a persistent living population.
```

Important final observations:

- direct maze-solving branches were successful in small MiniGrid tasks;
- `cyclic_motif_fast_replay` and `cyclic_operate_replay` remain the strongest
  practical historical baselines;
- target-side behavior priors can be extracted and executed;
- structural proxy metrics are not enough evidence of transfer;
- shuffled and random controls can produce misleading region-change signals;
- MultiRoom remained too sparse under the tested budgets;
- Dynamic-Obstacles became the best candidate for further transfer confirmation,
  but that confirmation was not pursued because the project direction changed.

## What Should Be Preserved

Useful pieces for future projects:

- MiniGrid adapter and state encoder patterns;
- versioned experiment discipline;
- negative-control discipline;
- behavior-prior and motif vocabulary;
- transfer-ladder calibration idea;
- documentation habit of writing interpretations into `docs/`.

Ideas to carry into the artificial-life project:

```text
experience should become inheritable structure
lineages should be first-class records
selection should explain why memory survives
benchmarks should be habitats, not the main goal
```

## What Should Not Be Carried Forward Unchanged

Do not use this repo as the direct architecture for the new project.

Avoid carrying over:

- benchmark-first framing;
- artifact-first pretraining evaluation;
- guard/matching-rule micro-optimization;
- large cyclic variant proliferation;
- final-score-only evaluation pressure.

The next project should start from:

```text
World
Habitat
Population
Organism
Genome
Lifetime
Experience
Inheritance
Lineage
Selection
EcologicalMemory
```

## Final Verification

After moving both existing projects to a shared virtual environment:

```text
uv run pytest
uv run pyright src scripts
uv run ruff check src scripts tests
```

Result:

```text
pytest: 46 passed
pyright: 0 errors
ruff: passed
```

The project environment now points at the shared playground environment:

```text
/Users/qlqwpy/Documents/游乐园/.venv
```

Local project symlink:

```text
maze_novelty/.venv -> ../.venv
```

## Future Use

If this repo is reopened, the best use is historical comparison or code reuse,
not continued optimization of the old transfer pipeline.

Recommended read order:

```text
docs/version_history.md
docs/v1.5.3_multiseed_pretraining_benefit_result.md
docs/v1.5.5_transfer_ladder_sweep_result.md
docs/theory_structural_novelty_and_options.md
```

The next active research line should be a new artificial-life project with
persistent population history as the primary object.
