from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression


@dataclass
class BlendConfig:
    residualize: bool = False


def residualize_y_on_X(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    model = LinearRegression()
    model.fit(X.values, y.values)
    resid = y - model.predict(X.values)
    return pd.Series(resid, index=y.index)


def train_quant_model(X147: pd.DataFrame, y: pd.Series, alpha: float = 0.001, l1_ratio: float = 0.1,
                      seed: int = 42) -> ElasticNet:
    """Train a lightweight ElasticNet quant model on 147-style features."""
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=2000)
    model.fit(X147.values, y.values)
    return model


def predict_quant_model(model: ElasticNet, X147: pd.DataFrame) -> np.ndarray:
    return model.predict(X147.values)


def train_blender(val_df: pd.DataFrame) -> LinearRegression:
    """Blender: rhat_blend = theta0 + theta1*rhat_text + theta2*rhat_quant trained on validation.

    Requires columns: rhat_text, rhat_quant, exret
    """
    required = {"rhat_text", "rhat_quant", "exret"}
    missing = required - set(val_df.columns)
    if missing:
        raise ValueError(f"Missing columns for blender training: {missing}")
    X = val_df[["rhat_text", "rhat_quant"]].values
    y = val_df["exret"].values
    reg = LinearRegression().fit(X, y)
    return reg


def predict_blender(model: LinearRegression, df: pd.DataFrame) -> np.ndarray:
    required = {"rhat_text", "rhat_quant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for blender prediction: {missing}")
    X = df[["rhat_text", "rhat_quant"]].values
    return model.predict(X)


def check_alignment(df_left: pd.DataFrame, df_right: pd.DataFrame, keys=("gvkey", "year_month")) -> None:
    if not keys:
        return
    left_keys = set(map(tuple, df_left[list(keys)].values))
    right_keys = set(map(tuple, df_right[list(keys)].values))
    if left_keys != right_keys:
        raise ValueError("Key alignment mismatch between frames on {keys}")
