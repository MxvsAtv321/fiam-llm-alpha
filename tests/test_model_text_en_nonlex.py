import pandas as pd
from fiam_llm.model_text_en import ExpandingWindowTextModel, ModelConfig


def test_nonlex_elasticnet_branch_scores():
    rows = []
    # create simple panel across 3 months
    for ym in ["2014-12", "2015-01", "2015-02"]:
        rows.append({"gvkey": "g1", "year_month": ym, "ret": 0.1 if ym=="2014-12" else -0.05, "x1": 1.0, "x2": 0.0})
        rows.append({"gvkey": "g2", "year_month": ym, "ret": -0.1 if ym=="2014-12" else 0.02, "x1": 0.0, "x2": 1.0})
    panel = pd.DataFrame(rows)
    model = ExpandingWindowTextModel(ModelConfig(model_type="elasticnet"))
    preds = model.fit_predict(panel, ["x1", "x2"], "ret")
    assert preds["score"].between(-1, 1).all()
    assert set(preds["year_month"]) == set(panel["year_month"])
