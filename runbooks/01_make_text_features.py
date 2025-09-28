from __future__ import annotations

import argparse
import polars as pl
from fiam_llm.io_loaders import align_text_to_month
from fiam_llm.features_text import build_features


def main(config_path: str) -> None:
    # Read cleaned sections and compute simple features
    in_path = "data/embeddings/sections_pca.parquet"
    out_path = "data/derived/text_features.parquet"
    df = pl.read_parquet(in_path)
    df = align_text_to_month(df, date_col="filing_date")
    feats = build_features(df, text_col="text_clean")
    feats.write_parquet(out_path)
    print(f"Wrote {out_path} with shape {feats.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
