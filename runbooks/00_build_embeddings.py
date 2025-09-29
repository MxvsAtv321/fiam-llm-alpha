from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import polars as pl
from fiam_llm.text_clean import normalize
from fiam_llm.embedder import DeterministicStubEmbedder, EmbedConfig, SectionEmbedder


def main(config_path: str) -> None:
    paths = {
        "in": "data/raw/filings/filings_small.csv",
        "out": "data/embeddings/sections_pca.parquet",
    }
    cfg = {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {"embedding": {"use_hf_embedder": False}}
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
    # For fixtures, derive gvkey from cik directly; in production join link table
    df = df.with_columns(pl.col("cik").alias("gvkey"))

    use_hf = bool(cfg.get("embedding", {}).get("use_hf_embedder", False))
    if use_hf:
        try:
            se = SectionEmbedder(model_name=cfg.get("embedding", {}).get("model_name", "ProsusAI/finbert"),
                                 pca_dim=cfg.get("embedding", {}).get("pca_dim", 64),
                                 cache_path=paths["out"])
            out_pdf = se.build_or_load_cache(df.select(["gvkey", "filing_date", "section", "text_clean"]).to_pandas())
            out = pl.from_pandas(out_pdf)
        except Exception:
            # Fallback to stub
            use_hf = False
    if not use_hf:
        embedder = DeterministicStubEmbedder(EmbedConfig(model_name="stub", max_tokens=1024, pca_dim=64))
        embedder.fit_pca(df["text_clean"].to_list())
        Z = embedder.transform(df["text_clean"].to_list())
        emb = pl.DataFrame(Z)
        out = pl.concat([df.select(["gvkey", "cik", "filing_date", "section", "text_clean"]), emb], how="horizontal")
        out.write_parquet(paths["out"])
    print(f"Wrote {paths['out']} with shape {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="config/defaults.yaml")
    args = ap.parse_args()
    main(args.config)
