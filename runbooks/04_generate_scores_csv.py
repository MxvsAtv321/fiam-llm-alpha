from __future__ import annotations

import argparse
import polars as pl


def main(config_path: str) -> None:
    path_in = "data/derived/text_scores.parquet"
    df = pl.read_parquet(path_in)
    df = df.select(["gvkey", "year_month", "mdna_sentiment_score", "risk_sentiment_score", "combined_score"])
    out_csv = "scores_for_portfolio.csv"
    df.write_csv(out_csv)
    print(f"Wrote {out_csv} with {df.shape[0]} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
