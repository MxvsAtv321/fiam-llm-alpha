from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class BlendConfig:
    residualize: bool = False


def residualize_y_on_X(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    model = LinearRegression()
    model.fit(X.values, y.values)
    resid = y - model.predict(X.values)
    return pd.Series(resid, index=y.index)


def blend_predictions(df: pd.DataFrame, ym_train: list[str], ym_val: list[str]) -> tuple[pd.DataFrame, dict]:
    # expects columns: gvkey, year_month, rhat_text, rhat_quant
    train_mask = df["year_month"].isin(ym_train + ym_val)
    te_mask = ~train_mask
    X = df.loc[train_mask, ["rhat_text", "rhat_quant"]].values
    y = df.loc[train_mask, "exret"].values
    reg = LinearRegression()
    reg.fit(X, y)
    df_out = df.copy()
    df_out["rhat_blend"] = reg.predict(df[["rhat_text", "rhat_quant"]].values)
    params = {"theta0": float(reg.intercept_), "theta1": float(reg.coef_[0]), "theta2": float(reg.coef_[1])}
    return df_out, params
