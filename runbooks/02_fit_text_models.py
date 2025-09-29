from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import mlflow

from fiam_llm.calibrate import cross_sectional_zscore, squashing_tanh, learn_composite_weights, combine_score


def main(config_path: str) -> None:
    feats = pl.read_parquet("data/derived/text_features.parquet")
    pdf = feats.to_pandas()
    pdf["tone"] = pdf.get("pos_1k", 0) - pdf.get("neg_1k", 0)
    pdf["z_tone"] = cross_sectional_zscore(pdf, "tone", by_col="year_month")
    pdf["s_mdna"] = squashing_tanh(pdf["z_tone"]).where(pdf["section"] == "MDNA")
    pdf["s_risk"] = (-squashing_tanh(pdf["z_tone"])) .where(pdf["section"] == "RISK")

    # Aggregate to firm-month per section
    mdna = pdf[pdf["section"] == "MDNA"].groupby(["gvkey", "year_month"])["s_mdna"].mean().rename("mdna_sentiment_score")
    risk = pdf[pdf["section"] == "RISK"].groupby(["gvkey", "year_month"])["s_risk"].mean().rename("risk_sentiment_score")
    sc = pd.concat([mdna, risk], axis=1).fillna(0.0)

    # Learn composite weights on 2013-2014 only
    mlflow.set_experiment("fiam_llm_alpha")
    Path("artifacts").mkdir(exist_ok=True)
    with mlflow.start_run(run_name="phase2_text_models"):
        mlflow.log_params({"composite_val_years": "2013-2014"})
        sc = sc.reset_index()
        sc["tone_delta"] = 0.0
        sc["novelty"] = 0.0
        sc["hedge_ratio"] = 0.0
        sc["exret_next"] = 0.0
        # For fixtures, we don't have returns here; keep zeros to test freeze mechanics only
        val_mask = sc["year_month"].between("2013-01", "2014-12")
        w = learn_composite_weights(sc.loc[val_mask])
        (Path("artifacts") / "composite_weights.json").write_text(json.dumps(w))
        mlflow.log_artifact("artifacts/composite_weights.json")

        # Apply frozen weights OOS 2015..end
        sc["combined_score"] = sc.apply(lambda r: combine_score(r, w), axis=1)
        out = pl.from_pandas(sc[["gvkey", "year_month", "mdna_sentiment_score", "risk_sentiment_score", "combined_score"]])
        out.write_parquet("data/derived/text_scores.parquet")
        print("Wrote data/derived/text_scores.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
