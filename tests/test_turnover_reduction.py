import pandas as pd
from fiam_llm.backtest import form_long_short


def test_turnover_reduction_with_band():
    cur = pd.DataFrame({"gvkey": [1, 2, 3, 4, 5], "score": [0.9, 0.7, 0.2, -0.5, -0.8]})
    nxt = pd.DataFrame({"gvkey": [1, 2, 3, 4, 5], "score": [0.88, 0.69, 0.21, -0.51, -0.79]})
    ls0 = form_long_short(cur, top_n=2, bottom_n=2, band=0.0)
    ls1 = form_long_short(nxt, top_n=2, bottom_n=2, band=0.0, prev=ls0)
    turn0 = 1 - len(ls0["long"] & ls1["long"]) / max(len(ls1["long"]), 1)
    ls1b = form_long_short(nxt, top_n=2, bottom_n=2, band=0.1, prev=ls0)
    turn1 = 1 - len(ls0["long"] & ls1b["long"]) / max(len(ls1b["long"]), 1)
    assert turn1 <= turn0
