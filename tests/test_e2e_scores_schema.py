import pandas as pd
from pathlib import Path


def test_scores_schema_and_bounds():
    p = Path("scores_for_portfolio.csv")
    assert p.exists(), "Run 00→05 pipeline first on fixtures."
    df = pd.read_csv(p)
    req = ["gvkey", "year_month", "mdna_sentiment_score", "risk_sentiment_score", "combined_score"]
    assert list(df.columns[:5]) == req, f"Schema must start with {req}"
    for col in ["mdna_sentiment_score", "risk_sentiment_score", "combined_score"]:
        assert df[col].between(-1, 1).all(), f"{col} must be in [-1,1]"
    assert not df[req].isna().any().any(), "No NaNs allowed in required columns"
