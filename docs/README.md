# FIAM LLM Alpha

![CI](https://github.com/MxvsAtv321/fiam-llm-alpha/actions/workflows/ci.yaml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

End-to-end pipeline to generate an LLM-powered sentiment score from 10-K/10-Q sections and backtest a long-short strategy.

## Quickstart

1. Create environment and install hooks:
```bash
make setup && make init
```
2. Run the pipeline on fixtures/synthetic and tests:
```bash
pytest --maxfail=1 --cov=src
make run-embeddings && make run-text && make run-train && make run-blend && make run-scores && make run-backtest
# Phase 3 per-period outputs and backtest from final weights
make run-numeric-blend && make run-risk-outputs && make run-backtest-final
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
 
## Data placement
- Place raw CSVs under `data/raw/` as per `DATA_DICTIONARY.md`.
- MLflow runs: local filesystem under `mlruns/`. Artifacts include `artifacts/composite_weights.json` and `reports/*`.
- Per-period CSVs: `reports/per_period/weights_after_risk_YYYY-MM.csv`, `returns_by_stock_YYYY-MM.csv`, `ret_cutout_YYYY-MM.csv`.
