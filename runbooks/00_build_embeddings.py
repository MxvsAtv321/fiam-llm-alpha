from __future__ import annotations

import argparse
import polars as pl
from fiam_llm.text_clean import normalize
from fiam_llm.embedder import DeterministicStubEmbedder, EmbedConfig


def main(config_path: str) -> None:
    # For tests, create a tiny synthetic embedding file based on available text
    # Expected input schema under data/raw/filings/*.csv: cik, filing_date, section, text
    # Here we just synthesize a small frame if none exists.
    paths = {
        "in": "data/raw/filings/filings_small.csv",
        "out": "data/embeddings/sections_pca.parquet",
    }
    try:
        df = pl.read_csv(paths["in"])  # synthetic fixture in tests
    except Exception:
        df = pl.DataFrame({
            "cik": ["0001"],
            "filing_date": ["2014-12-15"],
            "section": ["MDNA"],
            "text": ["We expect growth but there is risk."],
        })
    df = df.with_columns(pl.col("text").map_elements(normalize).alias("text_clean"))
    embedder = DeterministicStubEmbedder(EmbedConfig(model_name="stub", max_tokens=1024, pca_dim=64))
    embedder.fit_pca(df["text_clean"].to_list())
    Z = embedder.transform(df["text_clean"].to_list())
    emb = pl.DataFrame(Z)
    out = pl.concat([df.select(["cik", "filing_date", "section", "text_clean"]), emb], how="horizontal")
    out.write_parquet(paths["out"])
    print(f"Wrote {paths['out']} with shape {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
