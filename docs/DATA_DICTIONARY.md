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
