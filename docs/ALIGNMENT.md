# Time Alignment

- No forward-looking bias: only filings with `filing_date ≤ month_end(t)` are used for features at month t.
- `align_text_to_month` stamps `year_month` and `month_end`; downstream must enforce the filter.
- Predictions target `t+1` returns.
- Expanding windows: for year Y, train 2005..Y-2, validate Y-1..Y, predict Y.

## Leakage-proof sketch
- For each filing row, compute `year_month = YYYY-MM(filing_date)` and `month_end = month_end(filing_date_month)`.
- When building features for month t, filter filings: `filing_date ≤ month_end(t)`.
- Predictions are made at end of month t and evaluated on returns in t+1.
