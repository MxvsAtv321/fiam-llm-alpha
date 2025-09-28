from __future__ import annotations

import argparse
import polars as pl
import pandas as pd
from fiam_llm.calibrate import cross_sectional_zscore, squashing_tanh


def main(config_path: str) -> None:
    feats = pl.read_parquet("data/derived/text_features.parquet")
    # Build simple section-level scores: s_mdna = tanh(z(pos-neg)), s_risk = -tanh(z(neg))
    pdf = feats.to_pandas()
    pdf["tone"] = pdf.get("pos_1k", 0) - pdf.get("neg_1k", 0)
    # Cross-sectional z per month
    pdf["z_tone"] = cross_sectional_zscore(pdf, "tone", by_col="year_month")
    pdf["s_mdna"] = squashing_tanh(pdf["z_tone"]).where(pdf["section"] == "MDNA")
    pdf["s_risk"] = (-squashing_tanh(pdf["z_tone"])) .where(pdf["section"] == "RISK")
    # Aggregate to firm-month
    mdna = pdf[pdf["section"] == "MDNA"].groupby(["gvkey", "year_month"])["s_mdna"].mean().rename("mdna_sentiment_score")
    risk = pdf[pdf["section"] == "RISK"].groupby(["gvkey", "year_month"])["s_risk"].mean().rename("risk_sentiment_score")
    sc = pd.concat([mdna, risk], axis=1).fillna(0.0)
    sc["combined_score"] = (sc["mdna_sentiment_score"] + sc["risk_sentiment_score"]) / 2
    out = pl.from_pandas(sc.reset_index())
    out.write_parquet("data/derived/text_scores.parquet")
    print("Wrote data/derived/text_scores.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
