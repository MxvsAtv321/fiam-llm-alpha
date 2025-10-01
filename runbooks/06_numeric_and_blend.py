from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from fiam_llm.model_text_en import make_windows
from model_numeric import select_features, fit_numeric_model, predict_numeric_scores, scores_to_weights
from weights_blend import blend_weights, apply_banding, enforce_holdings, normalize_exposures


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    num_cfg = cfg.get("numeric_model", {})
    blend_cfg = cfg.get("blending", {})
    port_cfg = cfg.get("portfolio", {})
    paths = cfg.get("paths", {})

    # Load inputs
    text_scores = pd.read_parquet("data/derived/text_scores.parquet")
    # Returns panel fixture
    ret_path = Path("data/raw/ret_sample_small.csv")
    if not ret_path.exists():
        # Create tiny fixture if missing
        df = text_scores[["gvkey", "year_month"]].drop_duplicates().copy()
        rng = np.random.default_rng(0)
        df["exret_next"] = rng.normal(scale=0.01, size=len(df))
        for c in num_cfg.get("feature_whitelist", [])[:10]:
            df[c] = rng.normal(size=len(df))
        df.to_csv(ret_path, index=False)
    panel = pd.read_csv(ret_path)

    # Determine windows based on config
    wins = make_windows(train_start=cfg.get("universe", {}).get("train_start", "2005-01"),
                        first_oos_year=int(cfg.get("universe", {}).get("oos_start", "2015-01").split("-")[0]),
                        last_oos=cfg.get("universe", {}).get("oos_end", "2025-05"))

    # Build numeric model and blended weights per OOS year
    all_blended = []
    prev_month_df = None
    whitelist: List[str] = num_cfg.get("feature_whitelist", [])
    max_features = int(num_cfg.get("max_features", 30))
    stability_k = int(num_cfg.get("stability_k", 5))
    for w in wins:
        # Train+Val: from panel with feature columns available
        feat_cols = [c for c in whitelist if c in panel.columns]
        tv_mask = panel["year_month"].isin(w["train"] + w["val"]) if isinstance(w["val"], list) else panel["year_month"].isin(w["train"])  # guard
        tv = panel.loc[tv_mask].copy()
        if "exret_next" not in tv.columns:
            tv["exret_next"] = 0.0
        sel = select_features(tv, feat_cols, max_features=max_features, stability_k=stability_k)
        if not sel:
            sel = feat_cols[: min(5, len(feat_cols))]
        model = fit_numeric_model(tv, sel, target_col="exret_next")

        # Score OOS months for this year
        oos = panel.loc[panel["year_month"].isin(w["oos"])].copy()
        if oos.empty:
            continue
        num_raw = predict_numeric_scores(model, oos.assign(**{c: oos.get(c, pd.Series(0.0, index=oos.index)) for c in sel}), sel)
        num_w = scores_to_weights(num_raw)

        # Blend with LLM combined score
        llm = text_scores[text_scores["year_month"].isin(w["oos"])][["gvkey", "year_month", "combined_score"]].copy()
        blended = blend_weights(num_w, llm, numeric_weight=float(blend_cfg.get("numeric_weight", 0.8)), llm_weight=float(blend_cfg.get("llm_weight", 0.2)))

        # Apply banding and holdings enforcement + normalization month-by-month
        out_rows = []
        for ym, g in blended.groupby("year_month"):
            g = apply_banding(prev_month_df, g.copy(), band_fraction=float(cfg.get("portfolio", {}).get("band_fraction", 0.05)))
            g = enforce_holdings(g, min_holdings=int(cfg.get("portfolio", {}).get("min_holdings", 100)), max_holdings=int(cfg.get("portfolio", {}).get("max_holdings", 250)))
            g = normalize_exposures(g, target_long_abs=float(blend_cfg.get("target_long_abs", 1.0)), target_short_abs=float(blend_cfg.get("target_short_abs", 1.0)))
            out_rows.append(g.assign(year_month=ym))
            prev_month_df = g
        all_blended.append(pd.concat(out_rows, ignore_index=True))

    blended_all = pd.concat(all_blended, ignore_index=True) if all_blended else pd.DataFrame(columns=["gvkey", "year_month", "w_blend", "w_norm"])
    Path("data/derived").mkdir(parents=True, exist_ok=True)
    blended_all.to_parquet("data/derived/weights_blended.parquet", index=False)
    print("Wrote data/derived/weights_blended.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)


