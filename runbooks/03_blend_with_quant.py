from __future__ import annotations

import argparse
import polars as pl
import pandas as pd


def main(config_path: str) -> None:
    scores = pl.read_parquet("data/derived/text_scores.parquet").to_pandas()
    # For stub: set rhat_text = combined_score, rhat_quant = 0, rhat_blend = rhat_text
    preds = scores.rename(columns={"combined_score": "rhat_text"})
    preds["rhat_quant"] = 0.0
    preds["rhat_blend"] = preds["rhat_text"]
    out = pl.from_pandas(preds[["gvkey", "year_month", "rhat_text", "rhat_quant", "rhat_blend"]])
    out.write_parquet("data/derived/predictions.parquet")
    print("Wrote data/derived/predictions.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
