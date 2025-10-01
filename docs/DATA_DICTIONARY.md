# Data Dictionary

## Outputs
- `scores_for_portfolio.csv` columns:
  - `gvkey`: firm identifier
  - `year_month`: YYYY-MM
  - `mdna_sentiment_score`: [-1, 1]
  - `risk_sentiment_score`: [-1, 1]
  - `combined_score`: [-1, 1]

- `data/derived/text_scores.parquet`: same as above in parquet.
- `data/derived/predictions.parquet`: `{gvkey, year_month, rhat_text, rhat_quant, rhat_blend}`.

## Phase 3 per-period outputs
- `reports/per_period/weights_after_risk_YYYY-MM.csv`:
  - `{gvkey, year_month, weight}` after risk adjustment and normalization (long sum ≈ +1, short sum ≈ −1).
- `reports/per_period/returns_by_stock_YYYY-MM.csv`:
  - `{gvkey, year_month, exret}` realized one-month returns.
- `reports/per_period/ret_cutout_YYYY-MM.csv`:
  - Full return-panel cutout for that month (at least `{gvkey, year_month, exret}`; include `ret` if available).
