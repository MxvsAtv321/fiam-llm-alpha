import polars as pl
from fiam_llm.io_loaders import align_text_to_month


def test_align_text_to_month_no_forward_looking(synthetic_filings_csv):
    df = pl.read_csv(synthetic_filings_csv)
    out = align_text_to_month(df, date_col="filing_date")
    # Check year_month formatting
    assert out.select(pl.col("year_month").str.len_chars().max()).item() == 7
    # Ensure filing_date <= month_end(year_month)
    chk = out.with_columns(pl.col("filing_date").str.strptime(pl.Date, strict=False).alias("f"))
    assert (chk["f"] <= chk["month_end"]).all()
