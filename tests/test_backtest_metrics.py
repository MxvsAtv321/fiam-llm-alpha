import pandas as pd
from fiam_llm.backtest import compute_metrics


def test_compute_metrics_fields():
    pnl = pd.DataFrame({
        "year_month": ["2020-01", "2020-02", "2020-03"],
        "strategy_ret": [0.01, -0.02, 0.005],
    })
    mkt = pd.DataFrame({
        "year_month": ["2020-01", "2020-02", "2020-03"],
        "sp500_excess": [0.0, 0.0, 0.0],
    })
    met = compute_metrics(pnl, mkt)
    for k in ["alpha", "beta", "sharpe", "information_ratio", "max_drawdown", "worst_month"]:
        assert k in met
        assert isinstance(met[k], float)
