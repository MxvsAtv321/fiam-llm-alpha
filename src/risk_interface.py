from __future__ import annotations

from typing import Optional
import pandas as pd
from pathlib import Path


def load_risk_adjustments(year_month: str, pattern: str) -> Optional[pd.DataFrame]:
    """Load risk adjustments CSV for a given YYYY-MM using pattern like 'risk_adj_{YYYY}-{MM}.csv'.
    Returns None if file missing.
    """
    yyyy, mm = year_month.split("-")
    path = pattern.replace("{YYYY}", yyyy).replace("{MM}", mm)
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def apply_risk_adjustments(weights_month_df: pd.DataFrame, risk_df: Optional[pd.DataFrame], mode: str = "multiplicative") -> pd.DataFrame:
    """Apply risk adjustments and return re-normalized weights.

    weights_month_df: {gvkey, year_month, w_blend}
    risk_df: optional with columns {gvkey, adj_mult? adj_add?}
    mode: 'multiplicative' or 'additive'
    """
    if risk_df is None or risk_df.empty:
        return weights_month_df
    df = weights_month_df.merge(risk_df, on="gvkey", how="left")
    if mode == "multiplicative" and "adj_mult" in df.columns:
        df["w_blend"] = df["w_blend"] * df["adj_mult"].fillna(1.0)
    elif mode == "additive" and "adj_add" in df.columns:
        df["w_blend"] = df["w_blend"] + df["adj_add"].fillna(0.0)
    # Drop adj cols
    for c in ["adj_mult", "adj_add"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


