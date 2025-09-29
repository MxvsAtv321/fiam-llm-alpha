from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import mlflow

from fiam_llm.stack_blend import train_quant_model, predict_quant_model, train_blender, predict_blender


def main(config_path: str) -> None:
    scores = pl.read_parquet("data/derived/text_scores.parquet").to_pandas()
    preds = scores.rename(columns={"combined_score": "rhat_text"})[["gvkey", "year_month", "rhat_text"]]

    # Load a small quant fixture or generate synthetic columns for demo
    ret_path = Path("data/raw/ret_sample_small.csv")
    if ret_path.exists():
        ret = pd.read_csv(ret_path)
    else:
        base = preds[["gvkey", "year_month"]].copy()
        base["exret"] = 0.0
        base["f1"] = np.random.default_rng(0).normal(size=len(base))
        base["f2"] = np.random.default_rng(1).normal(size=len(base))
        ret = base
    # Align
    df = preds.merge(ret, on=["gvkey", "year_month"], how="left")
    fac_cols = [c for c in df.columns if c.startswith("f")]
    df[fac_cols] = df[fac_cols].fillna(0.0)
    df["exret"] = df["exret"].fillna(0.0)

    # Train quant on 2013-2014 validation, then blender on same years
    val_mask = df["year_month"].between("2013-01", "2014-12")
    oos_mask = df["year_month"] >= "2015-01"
    X_val = df.loc[val_mask, fac_cols]
    y_val = df.loc[val_mask, "exret"]
    q_model = train_quant_model(X_val, y_val)
    df["rhat_quant"] = predict_quant_model(q_model, df[fac_cols])
    b_model = train_blender(df.loc[val_mask, ["rhat_text", "rhat_quant", "exret"]])
    df["rhat_blend"] = predict_blender(b_model, df)

    out = df[["gvkey", "year_month", "rhat_text", "rhat_quant", "rhat_blend"]]
    pl.from_pandas(out).write_parquet("data/derived/predictions.parquet")
    print("Wrote data/derived/predictions.parquet")

    # MLflow logging
    mlflow.set_experiment("fiam_llm_alpha")
    with mlflow.start_run(run_name="phase2_blend"):
        mlflow.log_params({"quant_model": "ElasticNet", "fac_cols": len(fac_cols)})
        # simple val IC metric
        from scipy.stats import spearmanr
        ic_text = spearmanr(df.loc[val_mask, "rhat_text"], df.loc[val_mask, "exret"]).correlation or 0.0
        ic_blend = spearmanr(df.loc[val_mask, "rhat_blend"], df.loc[val_mask, "exret"]).correlation or 0.0
        mlflow.log_metric("val_ic_text", float(ic_text))
        mlflow.log_metric("val_ic_blend", float(ic_blend))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
