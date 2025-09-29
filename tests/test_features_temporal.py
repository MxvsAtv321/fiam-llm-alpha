import numpy as np
from fiam_llm.features_text import novelty, tone_delta


def test_novelty_monotonic():
    a = np.ones(8)
    b = np.ones(8)
    c = np.concatenate([np.ones(4), -np.ones(4)])
    n_ab = novelty(a, b)
    n_ac = novelty(a, c)
    assert 0.0 <= n_ab <= 2.0 and 0.0 <= n_ac <= 2.0
    assert n_ac >= n_ab


def test_tone_delta_zero_for_same():
    t = "profit growth"
    assert abs(tone_delta(t, t)) < 1e-6
