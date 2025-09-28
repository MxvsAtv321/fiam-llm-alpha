import pandas as pd
from fiam_llm.backtest import form_portfolio


def test_portfolio_holds_counts():
    scores = pd.DataFrame({
        "gvkey": [f"g{i:04d}" for i in range(300)],
        "year_month": ["2020-01"] * 300,
        "score": [i for i in range(300)],
    })
    port = form_portfolio(scores, top_n=100, bottom_n=100)
    one_month = port[port["year_month"] == "2020-01"]
    assert (one_month["weight"] > 0).sum() == 100
    assert (one_month["weight"] < 0).sum() == 100


def test_portfolio_graceful_small_universe():
    scores = pd.DataFrame({"gvkey": ["a", "b"], "year_month": ["2020-01", "2020-01"], "score": [1, -1]})
    port = form_portfolio(scores, top_n=100, bottom_n=100)
    one_month = port[port["year_month"] == "2020-01"]
    assert len(one_month) == 2
