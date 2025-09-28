from __future__ import annotations

import argparse
import os
import polars as pl
import pandas as pd
from fiam_llm.backtest import form_portfolio, backtest, compute_metrics


def main(config_path: str) -> None:
    os.makedirs("reports", exist_ok=True)
    scores = pl.read_parquet("data/derived/text_scores.parquet").to_pandas()
    # Use combined_score
    port = form_portfolio(scores.rename(columns={"combined_score": "score"})[["gvkey", "year_month", "score"]],
                          top_n=100, bottom_n=100)
    # synthetic returns: zero
    ret = scores[["gvkey", "year_month"]].copy()
    ret["exret"] = 0.0
    pnl = backtest(port, ret)
    mkt = pd.DataFrame({"year_month": pnl["year_month"], "sp500_excess": 0.0})
    metrics = compute_metrics(pnl, mkt)
    pl.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).write_csv("reports/metrics.csv")
    print("Backtest complete. Wrote reports/metrics.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
