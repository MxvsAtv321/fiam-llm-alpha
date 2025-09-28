import numpy as np
from fiam_llm.features_text import lm_counts_per_1k, novelty


def test_lm_counts_positive_negative():
    text_pos = "profit improve growth"
    text_neg = "risk loss uncertain"
    fpos = lm_counts_per_1k(text_pos)
    fneg = lm_counts_per_1k(text_neg)
    assert fpos["pos_1k"] > fpos["neg_1k"]
    assert fneg["neg_1k"] > fneg["pos_1k"]


def test_novelty_range():
    a = np.ones(8)
    b = np.ones(8)
    assert abs(novelty(a, b)) < 1e-6
    c = np.concatenate([np.ones(4), -np.ones(4)])
    n = novelty(a, c)
    assert n >= 0.0
