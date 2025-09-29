import polars as pl
from fiam_llm.io_loaders import write_parquet, read_parquet


def test_parquet_roundtrip(tmp_path):
    p = tmp_path / "x.parquet"
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    write_parquet(df, str(p))
    df2 = read_parquet(str(p))
    assert df2.shape == df.shape
    assert df2.dtypes == df.dtypes