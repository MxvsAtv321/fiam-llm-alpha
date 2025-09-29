import os
from pathlib import Path
import pandas as pd
from fiam_llm.reporting import export_scores_csv, plot_cumulative


def test_export_scores_csv(tmp_path):
    df = pd.DataFrame({
        "gvkey": ["1"],
        "year_month": ["2015-01"],
        "mdna_sentiment_score": [0.1],
        "risk_sentiment_score": [-0.2],
        "combined_score": [0.0],
    })
    out = tmp_path / "scores.csv"
    export_scores_csv(df, str(out))
    assert out.exists()
    df2 = pd.read_csv(out)
    req = ["gvkey", "year_month", "mdna_sentiment_score", "risk_sentiment_score", "combined_score"]
    assert list(df2.columns) == req
    for c in req[2:]:
        assert df2[c].between(-1, 1).all()


def test_plot_cumulative(tmp_path):
    pnl = pd.DataFrame({"year_month": ["2015-01", "2015-02"], "strategy_ret": [0.0, 0.0]})
    out = tmp_path / "curve.png"
    plot_cumulative(pnl, str(out))
    assert out.exists()
