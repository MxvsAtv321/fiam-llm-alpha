from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from fiam_llm.backtest import backtest, compute_metrics
from fiam_llm.reporting import plot_cumulative


def main() -> None:
    w = pd.read_parquet("data/derived/weights_final.parquet")
    r = pd.read_parquet("data/derived/returns_joined.parquet")
    # Compute monthly PnL from final weights
    df = w.merge(r, on=["gvkey", "year_month"], how="left").fillna({"exret": 0.0})
    pnl = df.assign(pnl=lambda x: x["weight"] * x["exret"]).groupby("year_month")["pnl"].sum().reset_index().rename(columns={"pnl": "strategy_ret"})
    # Benchmark (assume zero excess)
    mkt = pd.DataFrame({"year_month": pnl["year_month"], "sp500_excess": 0.0})
    metrics = compute_metrics(pnl, mkt)
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).to_csv("reports/metrics.csv", index=False)
    plot_cumulative(pnl, "reports/cumulative.png")
    print("Backtest from final weights complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    main()


