import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from fiam_llm.stack_blend import train_quant_model, predict_quant_model, train_blender, predict_blender


def test_blender_improves_ic_on_synthetic():
    rng = np.random.default_rng(0)
    n = 60
    rhat_text = rng.normal(size=n)
    f1 = rhat_text * 0.2 + rng.normal(scale=0.5, size=n)
    f2 = rng.normal(size=n)
    exret = 0.2 * rhat_text + 0.6 * f1 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"gvkey": range(n), "year_month": ["2013-01"] * n, "rhat_text": rhat_text, "f1": f1, "f2": f2, "exret": exret})
    q = train_quant_model(df[["f1", "f2"]], df["exret"])
    df["rhat_quant"] = predict_quant_model(q, df[["f1", "f2"]])
    b = train_blender(df[["rhat_text", "rhat_quant", "exret"]])
    df["rhat_blend"] = predict_blender(b, df)
    ic_text = spearmanr(df["rhat_text"], df["exret"]).correlation
    ic_blend = spearmanr(df["rhat_blend"], df["exret"]).correlation
    assert ic_blend >= ic_text - 1e-6
