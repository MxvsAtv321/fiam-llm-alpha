import pandas as pd
from fiam_llm.backtest import form_long_short


def test_turnover_banding_smoke():
    cur = pd.DataFrame({"gvkey": [1, 2, 3, 4, 5], "score": [0.9, 0.7, -0.6, -0.8, 0.2]})
    nxt = pd.DataFrame({"gvkey": [1, 2, 3, 4, 5], "score": [0.85, 0.68, -0.55, -0.82, 0.21]})
    ls0 = form_long_short(cur, top_n=2, bottom_n=2, band=0.05)
    ls1 = form_long_short(nxt, top_n=2, bottom_n=2, band=0.05, prev=ls0)
    assert len(ls1["long"]) == 2 and len(ls1["short"]) == 2
    assert len(set(ls0["long"]).intersection(ls1["long"])) >= 1
    assert len(set(ls0["short"]).intersection(ls1["short"])) >= 1
