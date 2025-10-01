from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


def _erfinv(y: np.ndarray) -> np.ndarray:
    a = 0.147
    s = np.sign(y)
    ln = np.log(1 - y ** 2)
    first = 2 / (np.pi * a) + ln / 2
    return s * np.sqrt(np.sqrt(first ** 2 - ln / a) - first)


def select_features(df: pd.DataFrame, whitelist: List[str], max_features: int, stability_k: int) -> List[str]:
    """Select robust features from whitelist using simple stability selection with ElasticNet.

    Only uses columns present in df; performs k trials with different seeds on the provided df
    (assumed to be train+val only), and keeps features that appear in at least ceil(k/2) fits,
    capped to max_features by descending frequency.
    """
    available = [c for c in whitelist if c in df.columns]
    if not available:
        return []
    X = df[available].fillna(0.0).values
    y = df["exret_next"].fillna(0.0).values
    counts = {c: 0 for c in available}
    for i in range(stability_k):
        seed = 42 + i
        model = ElasticNet(alpha=0.0005, l1_ratio=0.1, random_state=seed, max_iter=2000)
        model.fit(X, y)
        coef = np.abs(model.coef_)
        for c, w in zip(available, coef):
            if w != 0:
                counts[c] += 1
    thresh = ceil(stability_k / 2)
    ranked = sorted(available, key=lambda c: (-counts[c], c))
    selected = [c for c in ranked if counts[c] >= thresh][:max_features]
    return selected


def fit_numeric_model(train_df: pd.DataFrame, features: List[str], target_col: str = "exret_next") -> ElasticNet:
    """Fit a numeric ElasticNet model on selected features."""
    X = train_df[features].fillna(0.0).values
    y = train_df[target_col].fillna(0.0).values
    model = ElasticNet(alpha=0.0005, l1_ratio=0.1, random_state=42, max_iter=5000)
    model.fit(X, y)
    return model


def predict_numeric_scores(model: ElasticNet, oos_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Predict raw numeric scores for each row in oos_df using the trained model."""
    X = oos_df[features].fillna(0.0).values
    raw = model.predict(X)
    out = oos_df[["gvkey", "year_month"]].copy()
    out["score_raw"] = raw
    return out


def scores_to_weights(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Map cross-sectional scores to weights in [-1,1] per month via rank→z→tanh.

    Input columns: gvkey, year_month, score_raw
    Output columns: gvkey, year_month, weight_raw ∈ [-1,1]
    """
    df = monthly_df.copy()
    r = df.groupby("year_month")["score_raw"].rank(method="average", pct=True)
    z = np.sqrt(2) * _erfinv(2 * r.to_numpy() - 1)
    df["weight_raw"] = np.tanh(z)
    return df[["gvkey", "year_month", "weight_raw"]]


