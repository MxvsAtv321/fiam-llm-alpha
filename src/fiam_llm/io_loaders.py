from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import polars as pl


@dataclass(frozen=True)
class DataPaths:
    raw_dir: str
    derived_dir: str
    embeddings_dir: str


def read_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def read_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_parquet(df: pl.DataFrame, path: str) -> None:
    df.write_parquet(path)


def merge_cik_gvkey(filings: pl.DataFrame, link: pl.DataFrame) -> pl.DataFrame:
    return filings.join(link, on="cik", how="left")


def align_text_to_month(df: pl.DataFrame, date_col: str = "filing_date") -> pl.DataFrame:
    # Expect filing_date as YYYY-MM-DD
    out = df.with_columns(
        pl.col(date_col).str.strptime(pl.Date, strict=False).alias("_date"),
    ).with_columns(
        pl.col("_date").dt.strftime("%Y-%m").alias("year_month"),
        pl.col("_date").dt.month_end().alias("month_end"),
    )
    # Ensure no forward-looking: filing_date <= month_end(year_month)
    chk = out.with_columns(pl.col(date_col).str.strptime(pl.Date, strict=False).alias("_f"))
    assert (chk["_f"] <= chk["month_end"]).all(), "Time leakage: filing after month_end"
    return out.drop(["_date"])  # keep month_end for downstream checks


def month_filter(df: pl.DataFrame, ym: str) -> pl.DataFrame:
    return df.filter(pl.col("year_month") == ym)
