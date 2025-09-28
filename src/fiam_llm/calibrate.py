from __future__ import annotations

import numpy as np
import pandas as pd


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
