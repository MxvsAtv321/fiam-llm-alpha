from __future__ import annotations

from typing import Tuple
import os
from pathlib import Path
import pandas as pd
import numpy as np


def month_bounds(year_month: str) -> Tuple[str, str]:
    """Return period start and end YYYY-MM-DD for a given YYYY-MM."""
    p = pd.Period(year_month, freq="M")
    return (str(p.start_time.date()), str(p.end_time.date()))


def save_weights_after_risk(df_month: pd.DataFrame, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ym = df_month["year_month"].iloc[0]
    path = Path(out_dir) / f"weights_after_risk_{ym}.csv"
    df_month.to_csv(path, index=False)
    return str(path)


def save_returns_by_stock(df_month: pd.DataFrame, ret_panel: pd.DataFrame, out_dir: str) -> str:
    ym = df_month["year_month"].iloc[0]
    bystk = ret_panel[ret_panel["year_month"] == ym][["gvkey", "year_month", "exret"]].copy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"returns_by_stock_{ym}.csv"
    bystk.to_csv(path, index=False)
    return str(path)


def save_ret_cutout(year_month: str, ret_panel: pd.DataFrame, out_dir: str) -> str:
    cut = ret_panel[ret_panel["year_month"] == year_month].copy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"ret_cutout_{year_month}.csv"
    cut.to_csv(path, index=False)
    return str(path)


