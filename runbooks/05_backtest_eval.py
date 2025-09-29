from __future__ import annotations

import argparse
import os
import yaml
import polars as pl
import pandas as pd
from fiam_llm.backtest import form_portfolio, backtest, compute_metrics, form_long_short


def main(config_path: str) -> None:
    os.makedirs("reports", exist_ok=True)
    scores = pl.read_parquet("data/derived/text_scores.parquet").to_pandas()
    # Load band fraction from config
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {"portfolio": {"band_fraction": 0.0}}
    band = float(cfg.get("portfolio", {}).get("band_fraction", 0.0))

    # Use combined_score and record turnover
    monthly = scores.rename(columns={"combined_score": "score"})[["gvkey", "year_month", "score"]]
    monthly_groups = {ym: g.drop(columns=["year_month"]).copy() for ym, g in monthly.groupby("year_month")}
    prev = None
    turns = []
    port_rows = []
    for ym in sorted(monthly_groups.keys()):
        cur = monthly_groups[ym]
        ls = form_long_short(cur, top_n=100, bottom_n=100, band=band, prev=prev)
        prev = ls
        long_set, short_set = ls["long"], ls["short"]
        if len(long_set) > 0:
            turn = 0 if ym == sorted(monthly_groups.keys())[0] else 1 - len(long_set & long_prev) / max(len(long_set), 1)
        else:
            turn = 0.0
        long_prev = long_set
        turns.append({"year_month": ym, "turnover": turn})
        # Build equal-weight portfolio for this month
        cur_port = pd.DataFrame({"gvkey": list(long_set | short_set)})
        cur_port["year_month"] = ym
        cur_port["weight"] = cur_port["gvkey"].apply(lambda g: (1.0 / max(len(long_set), 1)) if g in long_set else (-1.0 / max(len(short_set), 1)))
        port_rows.append(cur_port)
    port = pd.concat(port_rows, ignore_index=True)
    # synthetic returns: zero
    ret = scores[["gvkey", "year_month"]].copy()
    ret["exret"] = 0.0
    pnl = backtest(port, ret)
    mkt = pd.DataFrame({"year_month": pnl["year_month"], "sp500_excess": 0.0})
    metrics = compute_metrics(pnl, mkt)
    met = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    met.to_csv("reports/metrics.csv", index=False)
    pd.DataFrame(turns).to_csv("reports/turnover.csv", index=False)
    print("Backtest complete. Wrote reports/metrics.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
