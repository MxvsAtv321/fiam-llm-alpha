from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import spearmanr

from .calibrate import cross_sectional_zscore, squashing_tanh


@dataclass
class ModelConfig:
    model_type: str  # 'lexicon_only', 'elasticnet'
    seed: int = 42


def rank_to_tanh(scores: pd.Series) -> pd.Series:
    # rank -> z -> tanh
    r = scores.rank(method="average", pct=True)
    z = pd.Series(np.sqrt(2) * erfinv(2 * r - 1), index=scores.index)
    return squashing_tanh(z)


def erfinv(y: pd.Series | np.ndarray) -> np.ndarray:
    # use approximation; but for tests we can rely on scipy? keep simple via numpy special
    from math import sqrt

    y = np.asarray(y, dtype=float)
    a = 0.147  # Winitzki approximation
    s = np.sign(y)
    ln = np.log(1 - y ** 2)
    first = 2 / (np.pi * a) + ln / 2
    return s * np.sqrt(np.sqrt(first ** 2 - ln / a) - first)


def _month_range(start: str, end: str) -> List[str]:
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def make_windows(train_start: str, first_oos_year: int, last_oos: str) -> List[dict]:
    """Create expanding windows per year.
    For OOS year Y: train = train_start..Y-2, val = Y-1..Y, oos = Y-01..Y-12 (bounded by last_oos).
    """
    last = pd.Period(last_oos, freq="M")
    windows: List[dict] = []
    for Y in range(first_oos_year, last.year + 1):
        train_end = f"{Y-2}-12"
        val_start = f"{Y-1}-01"
        val_end = f"{Y}-12"
        oos_start = f"{Y}-01"
        oos_end = f"{Y}-12"
        if Y == last.year:
            oos_end = str(last)
            val_end = str(min(pd.Period(f"{Y}-12", freq="M"), last))
        w = {
            "oos_year": Y,
            "train": _month_range(train_start, train_end),
            "val": _month_range(val_start, val_end),
            "oos": _month_range(oos_start, oos_end),
        }
        windows.append(w)
    return windows


class ExpandingWindowTextModel:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def _fit_en(self, X: np.ndarray, y: np.ndarray) -> ElasticNet:
        model = ElasticNet(random_state=self.cfg.seed)
        model.fit(X, y)
        return model

    def fit_predict(self, panel: pd.DataFrame, feature_cols: List[str], ret_col: str,
                    ym_col: str = "year_month", gvkey_col: str = "gvkey") -> pd.DataFrame:
        # panel contains year_month, gvkey, features, ret
        out_rows: List[Dict] = []
        months = sorted(panel[ym_col].unique())
        for ym in months:
            year = int(ym.split("-")[0])
            train_mask = panel[ym_col] < f"{year-1}-01"
            val_mask = (panel[ym_col] >= f"{year-1}-01") & (panel[ym_col] <= f"{year}-12")
            test_mask = panel[ym_col] == ym
            if self.cfg.model_type == "lexicon_only":
                # simple linear combination: pos - neg - hedge
                w = np.array([1.0 if "pos" in c else -1.0 if "neg" in c else -0.5 if "hedge" in c else 0.0 for c in feature_cols])
                yhat = panel.loc[test_mask, feature_cols].values @ w
            else:
                if train_mask.sum() < 20:
                    yhat = np.zeros(test_mask.sum())
                else:
                    Xtr = panel.loc[train_mask, feature_cols].values
                    ytr = panel.loc[train_mask, ret_col].values
                    model = self._fit_en(Xtr, ytr)
                    Xte = panel.loc[test_mask, feature_cols].values
                    yhat = model.predict(Xte)
            # map each month to [-1,1]
            s = pd.Series(yhat, index=panel.loc[test_mask].index)
            r = s.rank(method="average", pct=True)
            z = pd.Series(np.sqrt(2) * erfinv(2 * r - 1), index=s.index)
            out = squashing_tanh(z)
            tmp = panel.loc[test_mask, [gvkey_col, ym_col]].copy()
            tmp["score"] = out.values
            out_rows.append(tmp)
        return pd.concat(out_rows, ignore_index=True)
