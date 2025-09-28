from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import polars as pl

POS = {"gain", "improve", "growth", "profit"}
NEG = {"loss", "decline", "risk", "uncertain"}
HEDGE = {"may", "might", "could", "approximately"}
FWD = {"expect", "forecast", "plan", "anticipate"}


def _count_terms(text: str, vocab: set[str]) -> int:
    tokens = text.lower().split()
    return sum(1 for t in tokens if t in vocab)


def lm_counts_per_1k(text: str) -> Dict[str, float]:
    tokens = text.lower().split()
    n = max(len(tokens), 1)
    scale = 1000.0 / n
    return {
        "pos_1k": _count_terms(text, POS) * scale,
        "neg_1k": _count_terms(text, NEG) * scale,
        "hedge_1k": _count_terms(text, HEDGE) * scale,
        "fwd_1k": _count_terms(text, FWD) * scale,
    }


def readability_fog(text: str) -> float:
    sentences = max(text.count("."), 1)
    words = max(len(text.split()), 1)
    complex_words = sum(1 for w in text.split() if len(w) >= 7)
    asl = words / sentences
    pclw = complex_words / words * 100
    return 0.4 * (asl + pclw)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def novelty(emb_t: np.ndarray, emb_prev: np.ndarray) -> float:
    return 1.0 - cosine_similarity(emb_t, emb_prev)


def build_features(df: pl.DataFrame, text_col: str = "text_clean") -> pl.DataFrame:
    feats = []
    for row in df.iter_rows(named=True):
        features = lm_counts_per_1k(row[text_col])
        features["fog"] = readability_fog(row[text_col])
        features["gvkey"] = row["gvkey"]
        features["year_month"] = row["year_month"]
        features["section"] = row.get("section", "UNKNOWN")
        feats.append(features)
    return pl.DataFrame(feats)
