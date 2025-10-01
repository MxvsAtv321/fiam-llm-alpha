import numpy as np
import pandas as pd
from model_numeric import select_features, fit_numeric_model, predict_numeric_scores, scores_to_weights


def test_numeric_feature_selection_and_weights():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "exret_next": rng.normal(size=50),
        "a": rng.normal(size=50),
        "b": rng.normal(size=50),
        "c": rng.normal(size=50),
    })
    feats = select_features(df, ["a", "b", "c", "d"], max_features=2, stability_k=3)
    assert set(feats).issubset({"a", "b", "c"})
    m = fit_numeric_model(df, feats)
    oos = pd.DataFrame({"gvkey": ["g1", "g2"], "year_month": ["2015-01", "2015-01"], **{f: [0.1, -0.1] for f in feats}})
    raw = predict_numeric_scores(m, oos, feats)
    w = scores_to_weights(raw)
    assert w["weight_raw"].between(-1, 1).all()


