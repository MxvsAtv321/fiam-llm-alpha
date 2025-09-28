from __future__ import annotations

import pandas as pd
import numpy as np


def form_long_short(df_month: pd.DataFrame, top_n: int = 100, bottom_n: int = 100,
                    band: float = 0.0, prev: dict | None = None) -> dict:
    """
    band: percentile band in [0,1]; if >0, retain prev names within band to reduce churn.
    prev: {"long": set([...]), "short": set([...])}
    Returns dict with sets for long and short.
    """
    df = df_month.sort_values("score", ascending=False).reset_index(drop=True)
    if band > 0 and prev:
        hi = df["score"].quantile(1 - band)
        lo = df["score"].quantile(band)
        hi_set = set(df.loc[df["score"] >= hi, "gvkey"])
        lo_set = set(df.loc[df["score"] <= lo, "gvkey"])
        keep_long = [g for g in prev.get("long", set()) if g in hi_set]
        keep_short = [g for g in prev.get("short", set()) if g in lo_set]
    else:
        keep_long, keep_short = [], []
    long = list({*keep_long, *df["gvkey"].head(top_n).tolist()})[:top_n]
    short = list({*keep_short, *df["gvkey"].tail(bottom_n).tolist()})[:bottom_n]
    return {"long": set(long), "short": set(short)}


def form_portfolio(scores: pd.DataFrame, top_n: int = 100, bottom_n: int = 100,
                   score_col: str = "score") -> pd.DataFrame:
    out = []
    for ym, grp in scores.groupby("year_month"):
        g_sorted = grp.sort_values(score_col, ascending=False)
        long = g_sorted.head(top_n)
        short = g_sorted.tail(bottom_n)
        long_w = 1.0 / max(len(long), 1)
        short_w = -1.0 / max(len(short), 1)
        tmp = pd.concat([
            long.assign(weight=long_w),
            short.assign(weight=short_w),
        ])
        tmp["year_month"] = ym
        out.append(tmp[["gvkey", "year_month", "weight"]])
    return pd.concat(out, ignore_index=True)


def backtest(port: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    # returns: gvkey, year_month, exret
    df = port.merge(returns, on=["gvkey", "year_month"], how="left").fillna({"exret": 0.0})
    pnl = df.assign(pnl=lambda x: x["weight"] * x["exret"]).groupby("year_month")["pnl"].sum().reset_index()
    pnl.rename(columns={"pnl": "strategy_ret"}, inplace=True)
    return pnl


def compute_metrics(pnl: pd.DataFrame, mkt: pd.DataFrame) -> dict:
    df = pnl.merge(mkt[["year_month", "sp500_excess"]], on="year_month", how="left").fillna(0.0)
    r = df["strategy_ret"].values
    m = df["sp500_excess"].values
    alpha = r.mean() - m.mean()
    beta = np.cov(r, m)[0, 1] / (np.var(m) + 1e-8)
    sharpe = r.mean() / (r.std(ddof=0) + 1e-8)
    ir = (r - m).mean() / ((r - m).std(ddof=0) + 1e-8)
    cum = (1 + r).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum / peak - 1).min()
    worst = r.min()
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "sharpe": float(sharpe),
        "information_ratio": float(ir),
        "max_drawdown": float(dd),
        "worst_month": float(worst),
    }
