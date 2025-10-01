from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def blend_weights(numeric_w: pd.DataFrame, llm_scores: pd.DataFrame, numeric_weight: float, llm_weight: float) -> pd.DataFrame:
    """Blend numeric weights and llm combined scores into a single column w_blend.

    Missing values are treated as zero. Expects columns:
    - numeric_w: {gvkey, year_month, weight_raw}
    - llm_scores: {gvkey, year_month, combined_score}
    """
    n = numeric_w.rename(columns={"weight_raw": "w_num"})
    l = llm_scores[["gvkey", "year_month", "combined_score"]].rename(columns={"combined_score": "w_llm"})
    df = n.merge(l, on=["gvkey", "year_month"], how="outer").fillna({"w_num": 0.0, "w_llm": 0.0})
    df["w_blend"] = numeric_weight * df["w_num"] + llm_weight * df["w_llm"]
    return df[["gvkey", "year_month", "w_blend"]]


def enforce_holdings(df_month: pd.DataFrame, min_holdings: int, max_holdings: int) -> pd.DataFrame:
    """Keep names with largest absolute weights, bounded by [min_holdings, max_holdings]."""
    k = min(max(len(df_month), min_holdings), max_holdings)
    return df_month.reindex(df_month["w_blend"].abs().sort_values(ascending=False).head(k).index)


def _renorm_side(weights: pd.Series, target_abs: float, sign: int) -> pd.Series:
    mask = (weights * sign) > 0
    s = weights[mask].abs().sum()
    if s == 0:
        return weights
    weights.loc[mask] = weights.loc[mask] * (target_abs / s)
    return weights


def normalize_exposures(df_month: pd.DataFrame, target_long_abs: float = 1.0, target_short_abs: float = 1.0) -> pd.DataFrame:
    """Normalize so sum of longs equals target_long_abs and shorts equals -target_short_abs."""
    out = df_month.copy()
    out["w_norm"] = out["w_blend"].copy()
    out["w_norm"] = _renorm_side(out["w_norm"], target_long_abs, sign=+1)
    out["w_norm"] = _renorm_side(out["w_norm"], target_short_abs, sign=-1)
    return out


def apply_banding(prev: pd.DataFrame | None, current: pd.DataFrame, band_fraction: float) -> pd.DataFrame:
    """Reduce turnover by retaining previous holdings within percentile bands; then reapply blend weights.

    If prev is None, return current as-is.
    """
    if prev is None or band_fraction <= 0:
        return current
    hi = current["w_blend"].quantile(1 - band_fraction)
    lo = current["w_blend"].quantile(band_fraction)
    keep = set(prev["gvkey"]) & set(current.loc[(current["w_blend"] >= hi) | (current["w_blend"] <= lo), "gvkey"]) 
    cur_sorted = current.sort_values("w_blend", ascending=False)
    # Retain incumbents within band, then fill by strongest weights
    kept = cur_sorted[cur_sorted["gvkey"].isin(keep)]
    rest = cur_sorted[~cur_sorted["gvkey"].isin(keep)]
    return pd.concat([kept, rest], ignore_index=True)


