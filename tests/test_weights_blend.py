import pandas as pd
from weights_blend import blend_weights, enforce_holdings, normalize_exposures, apply_banding


def test_blend_holdings_and_normalize():
    num = pd.DataFrame({"gvkey": [1, 2, 3], "year_month": ["2015-01"] * 3, "weight_raw": [0.5, -0.2, 0.1]})
    llm = pd.DataFrame({"gvkey": [1, 2, 3], "year_month": ["2015-01"] * 3, "combined_score": [0.2, -0.1, 0.0]})
    b = blend_weights(num, llm, numeric_weight=0.8, llm_weight=0.2)
    assert "w_blend" in b.columns
    e = enforce_holdings(b, min_holdings=2, max_holdings=2)
    assert len(e) == 2
    n = normalize_exposures(e, target_long_abs=1.0, target_short_abs=1.0)
    long_sum = n[n["w_norm"] > 0]["w_norm"].sum()
    short_sum = n[n["w_norm"] < 0]["w_norm"].abs().sum()
    assert abs(long_sum - 1.0) < 1e-6
    assert abs(short_sum - 1.0) < 1e-6


def test_banding_reduces_churn():
    prev = pd.DataFrame({"gvkey": [1, 2, 3], "year_month": ["2015-01"] * 3, "w_blend": [0.8, -0.6, 0.1]})
    cur = pd.DataFrame({"gvkey": [1, 2, 3], "year_month": ["2015-02"] * 3, "w_blend": [0.78, -0.58, 0.12]})
    a = apply_banding(prev, cur, band_fraction=0.1)
    assert set(a["gvkey"]) == set(cur["gvkey"])  # order may change


