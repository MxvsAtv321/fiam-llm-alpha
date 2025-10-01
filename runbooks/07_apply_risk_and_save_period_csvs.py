from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

from risk_interface import load_risk_adjustments, apply_risk_adjustments
from period_io import save_weights_after_risk, save_returns_by_stock, save_ret_cutout


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    risk_cfg = cfg.get("risk", {})
    out_dir = cfg.get("per_period_outputs", {}).get("dir", "reports/per_period")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    blended = pd.read_parquet("data/derived/weights_blended.parquet")
    ret_panel = pd.read_csv("data/raw/ret_sample_small.csv") if Path("data/raw/ret_sample_small.csv").exists() else blended[["gvkey", "year_month"]].assign(exret=0.0)

    weights_final = []
    returns_joined = []
    for ym, g in blended.groupby("year_month"):
        risk_path = risk_cfg.get("file_pattern", "").replace("{YYYY}", ym.split("-")[0]).replace("{MM}", ym.split("-")[1]) if risk_cfg else ""
        risk_df = load_risk_adjustments(ym, risk_cfg.get("file_pattern", "")) if risk_cfg else None
        g2 = apply_risk_adjustments(g[["gvkey", "year_month", "w_norm"]].rename(columns={"w_norm": "weight"}), risk_df, mode=risk_cfg.get("mode", "multiplicative"))
        save_weights_after_risk(g2, out_dir)
        save_returns_by_stock(g2, ret_panel, out_dir)
        save_ret_cutout(ym, ret_panel, out_dir)
        weights_final.append(g2)
        returns_joined.append(ret_panel[ret_panel["year_month"] == ym])

    w_final = pd.concat(weights_final, ignore_index=True) if weights_final else pd.DataFrame(columns=["gvkey", "year_month", "weight"])
    r_join = pd.concat(returns_joined, ignore_index=True) if returns_joined else pd.DataFrame(columns=["gvkey", "year_month", "exret"])
    Path("data/derived").mkdir(parents=True, exist_ok=True)
    w_final.to_parquet("data/derived/weights_final.parquet", index=False)
    r_join.to_parquet("data/derived/returns_joined.parquet", index=False)
    print("Wrote weights_final and returns_joined")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)


