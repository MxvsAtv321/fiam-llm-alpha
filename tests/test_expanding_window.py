import pandas as pd
from fiam_llm.model_text_en import ExpandingWindowTextModel, ModelConfig


def test_expanding_window_splits():
    # Create a small panel with months across 2013-2015
    rows = []
    for ym in ["2013-12", "2014-01", "2014-12", "2015-01"]:
        rows.append({"gvkey": "0001", "year_month": ym, "ret": 0.0, "pos_1k": 0.1, "neg_1k": 0.0, "hedge_1k": 0.0})
    panel = pd.DataFrame(rows)
    model = ExpandingWindowTextModel(ModelConfig(model_type="lexicon_only"))
    preds = model.fit_predict(panel, ["pos_1k", "neg_1k", "hedge_1k"], "ret")
    # Should produce a score for each test month in panel (all months here)
    assert set(preds["year_month"]) == set(panel["year_month"])
