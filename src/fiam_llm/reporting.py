from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def export_scores_csv(df: pd.DataFrame, path: str) -> None:
    cols = ["gvkey", "year_month", "mdna_sentiment_score", "risk_sentiment_score", "combined_score"]
    df[cols].to_csv(path, index=False)


def plot_cumulative(pnl: pd.DataFrame, out_path: str) -> None:
    cum = (1 + pnl["strategy_ret"]).cumprod()
    plt.figure(figsize=(8, 4))
    plt.plot(pnl["year_month"], cum)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
