Guardrails for FIAM LLM Alpha

- NEVER use information from t+1 or later when constructing features at month t.
- All functions should be pure; pass explicit date arguments.
- Prefer parquet and polars for I/O; batch operations only.
- Cache embeddings; PCA reduce to 64–128 dims; store float32.
- Tests must cover all date alignment logic.
- Public functions: include type hints, docstrings, and unit tests.
- Reproducibility: deterministic seeds across numpy/torch/sklearn; MLflow logging.
- Interoperability: emit CSV keyed by `{gvkey, year_month}` with columns `mdna_sentiment_score, risk_sentiment_score, combined_score`.
- Quality gates: ruff, black, mypy; pytest coverage ≥ 85%.
