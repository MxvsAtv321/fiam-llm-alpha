import pandas as pd
import numpy as np
from fiam_llm.calibrate import cross_sectional_zscore, squashing_tanh


def test_zscore_and_tanh_bounds():
    df = pd.DataFrame({"year_month": ["2020-01"] * 5, "v": [1, 2, 3, 4, 5]})
    z = cross_sectional_zscore(df, "v", by_col="year_month")
    assert abs(z.mean()) < 1e-6
    s = squashing_tanh(z)
    assert (s.values <= 1.0 + 1e-6).all()
    assert (s.values >= -1.0 - 1e-6).all()
