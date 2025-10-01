import pandas as pd
from period_io import month_bounds, save_weights_after_risk, save_returns_by_stock, save_ret_cutout


def test_month_bounds_and_csvs(tmp_path):
    start, end = month_bounds("2015-01")
    assert start.endswith("-01") and end.endswith("-31")
    w = pd.DataFrame({"gvkey": [1], "year_month": ["2015-01"], "weight": [0.5]})
    r = pd.DataFrame({"gvkey": [1], "year_month": ["2015-01"], "exret": [0.01]})
    a = save_weights_after_risk(w, str(tmp_path))
    b = save_returns_by_stock(w, r, str(tmp_path))
    c = save_ret_cutout("2015-01", r, str(tmp_path))
    assert tmp_path.joinpath(a).exists() and tmp_path.joinpath(b).exists() and tmp_path.joinpath(c).exists()


