# Market Benchmark Contract

## Purpose

Market experiments should produce comparable outputs even when they run in a
separate project. The goal is to compare mechanisms, not copy maze metrics
literally.

## Required Run Metadata

Each market run should report:

```text
experiment_name
environment_id
environment_version
method
seed
generation
runtime_seconds
time_limited
```

## Required Performance Metrics

Use these fields when available:

```text
score
profit
drawdown
sharpe_like
sortino_like
turnover
trade_count
win_rate
catastrophic_loss_rate
regime_success_rate
```

`score` should be the method's primary comparison metric. It can be a weighted
market score, but the component metrics above should still be exported.

## Required Mechanism Metrics

For cyclic/novelty methods, export:

```text
phase
motif_count
fast_motif_count
medium_motif_count
candidate_motif_count
niche_count
replay_bank_size
stress_pass_rate
motif_bank_fill_generation
replay_improvement_delta
```

If a metric does not apply, write `0` rather than omitting the column.

## Market-Specific Motif Meaning

Maze fast success maps poorly to market tasks. Market motif quality should use
risk-adjusted local behavior:

```text
local_profit
local_drawdown
local_volatility
profit_to_drawdown
holding_duration
turnover
regime_id
regime_transition_survival
```

Fast completion is only one possible quality dimension. In market tasks, a motif
with low return but very low drawdown may be valuable.

## Export Format

Preferred format:

```text
CSV with one row per method/generation/seed
```

Minimum columns:

```text
experiment_name,environment_id,environment_version,method,seed,generation,
phase,score,profit,drawdown,sharpe_like,turnover,motif_count,niche_count,
replay_bank_size,stress_pass_rate,runtime_seconds,time_limited
```

## Interpretation Rule

Do not treat more motifs or more niches as proof of improvement. They are
mechanism diagnostics. Final judgment should still include score, drawdown,
stability across regimes, and catastrophic-loss behavior.
