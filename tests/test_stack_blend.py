import numpy as np
import pandas as pd
from fiam_llm.stack_blend import residualize_y_on_X
from sklearn.linear_model import LinearRegression


def test_residualization_and_blender():
    rng = np.random.default_rng(0)
    n = 30
    rhat_text = rng.normal(size=n)
    rhat_quant = rhat_text * 0.5 + rng.normal(scale=0.1, size=n)
    y = 0.3 * rhat_text + 0.6 * rhat_quant + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"rhat_text": rhat_text, "rhat_quant": rhat_quant, "exret": y})
    resid = residualize_y_on_X(df["exret"], df[["rhat_text", "rhat_quant"]])
    assert abs(resid.mean()) < 1e-6
    m = LinearRegression().fit(df[["rhat_text", "rhat_quant"]].values, df["exret"].values)
    assert np.any(np.abs(m.coef_) > 1e-6)
