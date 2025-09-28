from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def cross_sectional_zscore(df: pd.DataFrame, value_col: str, by_col: str) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros_like(x), index=x.index)
        return (x - mu) / sd

    return df.groupby(by_col)[value_col].transform(_z)


def squashing_tanh(x: pd.Series, clip: float = 3.0) -> pd.Series:
    x_clipped = x.clip(-clip, clip)
    return np.tanh(x_clipped)


def learn_composite_weights(val_df: pd.DataFrame) -> dict:
    """
    val_df columns: s_mdna, s_risk, tone_delta, novelty, hedge_ratio, exret_next
    Returns dict with keys alpha..eta.
    """
    cols = ["s_mdna", "s_risk", "tone_delta", "novelty", "hedge_ratio"]
    X = val_df[cols].values
    y = val_df["exret_next"].values
    mdl = Ridge(alpha=1.0, random_state=42).fit(X, y)
    w = dict(zip(["alpha", "beta", "gamma", "delta", "eta"], mdl.coef_.tolist()))
    return w


def combine_score(row: pd.Series, w: dict) -> float:
    z = (
        w.get("alpha", 1.0) * row.get("s_mdna", 0.0)
        + w.get("beta", 1.0) * row.get("s_risk", 0.0)
        + w.get("gamma", 0.0) * row.get("tone_delta", 0.0)
        + w.get("delta", 0.0) * row.get("novelty", 0.0)
        + w.get("eta", 0.0) * row.get("hedge_ratio", 0.0)
    )
    return float(np.tanh(z))
