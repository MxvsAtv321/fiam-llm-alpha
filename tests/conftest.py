import os
import polars as pl
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_dirs(tmp_path_factory):
    for d in ["data/raw", "data/derived", "data/embeddings", "tests/fixtures"]:
        os.makedirs(d, exist_ok=True)


@pytest.fixture(scope="session")
def synthetic_filings_csv(tmp_path_factory):
    path = "data/raw/filings/filings_small.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "cik": ["0001", "0001", "0002", "0002"],
        "filing_date": ["2014-12-15", "2015-01-20", "2014-12-28", "2015-02-10"],
        "section": ["MDNA", "RISK", "MDNA", "RISK"],
        "text": ["We expect growth.", "There is risk and uncertainty.", "Profit improve.", "Loss might occur."]
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def synthetic_returns_csv():
    path = "data/raw/ret_sample_small.csv"
    df = pd.DataFrame({
        "gvkey": ["0001", "0001", "0002", "0002"],
        "year_month": ["2014-12", "2015-01", "2014-12", "2015-02"],
        "exret": [0.01, -0.02, 0.03, 0.00]
    })
    df.to_csv(path, index=False)
    return path
