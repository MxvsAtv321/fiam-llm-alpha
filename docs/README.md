# FIAM LLM Alpha

End-to-end pipeline to generate an LLM-powered sentiment score from 10-K/10-Q sections and backtest a long-short strategy.

## Quickstart

1. Create environment and install hooks:
```bash
make setup
```
2. Initialize git-lfs and DVC (optional):
```bash
make init
```
3. Run the pipeline on fixtures/synthetic:
```bash
make run-embeddings
make run-text
make run-train
make run-blend
make run-scores
make run-backtest
```

Artifacts:
- `data/embeddings/sections_pca.parquet`
- `data/derived/text_features.parquet`
- `data/derived/text_scores.parquet`
- `data/derived/predictions.parquet`
- `scores_for_portfolio.csv`
- `reports/metrics.csv`

## Reproducibility
- Seeds set via config; parquet I/O; deterministic stubs for CI.
- Pre-commit: ruff, black, mypy; CI enforces ≥85% coverage.
