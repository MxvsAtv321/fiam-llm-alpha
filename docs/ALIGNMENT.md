# Time Alignment

- No forward-looking bias: only filings with `filing_date ≤ month_end(t)` are used for features at month t.
- `align_text_to_month` stamps `year_month` and `month_end`; downstream must enforce the filter.
- Predictions target `t+1` returns.
- Expanding windows: for year Y, train 2005..Y-2, validate Y-1..Y, predict Y.
