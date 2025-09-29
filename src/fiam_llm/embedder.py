from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class EmbedConfig:
    model_name: str
    max_tokens: int
    pca_dim: int
    seed: int = 42


class DeterministicStubEmbedder:
    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.pca: PCA | None = None
        self._effective_dim: int | None = None

    def _tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        return tokens[: self.cfg.max_tokens]

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(256, dtype=np.float32)
        for t in tokens:
            h = abs(hash(t)) % 256
            vec[h] += 1.0
        if vec.sum() > 0:
            vec = vec / np.linalg.norm(vec)
        return vec

    def fit_pca(self, texts: Iterable[str]) -> None:
        X = np.stack([self.embed_text(t) for t in texts])
        n_comp = int(min(self.cfg.pca_dim, X.shape[0], X.shape[1]))
        if n_comp <= 0:
            n_comp = 1
        self._effective_dim = n_comp
        self.pca = PCA(n_components=n_comp, random_state=self.cfg.seed)
        self.pca.fit(X)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        X = np.stack([self.embed_text(t) for t in texts])
        if self.pca is None:
            # Fallback: truncate or pad to desired dim
            Z = X[:, : min(self.cfg.pca_dim, X.shape[1])]
        else:
            Z = self.pca.transform(X)
        # Pad to cfg.pca_dim if needed
        target = self.cfg.pca_dim
        if Z.shape[1] < target:
            pad = np.zeros((Z.shape[0], target - Z.shape[1]), dtype=Z.dtype)
            Z = np.concatenate([Z, pad], axis=1)
        return Z.astype(np.float32)


# Optional FinBERT-compatible embedder with cache
from pathlib import Path
import pandas as pd
import contextlib

try:  # deferred import
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:  # pragma: no cover
    AutoTokenizer = AutoModel = None
    torch = None


class SectionEmbedder:
    def __init__(self, model_name: str, pca_dim: int = 64, cache_path: str = "data/embeddings/sections_pca.parquet"):
        self.model_name = model_name
        self.pca_dim = pca_dim
        self.cache_path = Path(cache_path)
        self._tok = None
        self._mdl = None
        self._pca = PCA(n_components=min(pca_dim, 64), random_state=42)
        self._effective_dim = None

    def _load_hf(self) -> None:
        assert AutoTokenizer is not None, "Transformers not installed."
        if self._tok is None or self._mdl is None:  # pragma: no cover
            self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self._mdl = AutoModel.from_pretrained(self.model_name)
            self._mdl.eval()
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                self._mdl.cuda()

    def _encode_chunks(self, texts: Iterable[str], max_len: int = 512) -> np.ndarray:
        self._load_hf()
        outs: List[np.ndarray] = []
        for t in texts:  # pragma: no cover (heavy path not covered in CI)
            toks = self._tok(t, truncation=False, add_special_tokens=False)["input_ids"]
            vecs: List[np.ndarray] = []
            for i in range(0, len(toks), max_len):
                chunk = toks[i : i + max_len]
                if not chunk:
                    break
                enc = self._tok.encode_plus(chunk, is_split_into_words=False, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_len)
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                if torch is not None and hasattr(torch, "no_grad"):
                    ctx = torch.no_grad()
                    with ctx:
                        out = self._mdl(**enc).last_hidden_state  # [1, L, H]
                        attn = enc["attention_mask"].float()  # [1, L]
                        w = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
                        pooled = (out * w.unsqueeze(-1)).sum(dim=1)  # [1, H]
                        vecs.append(pooled.squeeze(0).detach().cpu().numpy())
                else:
                    # Numpy fallback for mocked tests
                    out = self._mdl(**enc).last_hidden_state  # DummyTensor
                    attn = enc["attention_mask"].float()
                    attn_np = attn.numpy()
                    den_np = attn.sum(dim=1, keepdim=True).numpy()
                    w_np = attn_np / (den_np + 1e-8)
                    out_np = out.numpy()  # [1, L, H]
                    pooled_np = (out_np * w_np[..., None]).sum(axis=1)  # [1, H]
                    vecs.append(pooled_np.squeeze(0))
            if len(vecs) == 0 and self._mdl is not None:
                outs.append(np.zeros(self._mdl.config.hidden_size, dtype=np.float32))
            else:
                outs.append(np.stack(vecs, 0).mean(0).astype(np.float32))
        return np.stack(outs, 0)

    def fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        n_comp = int(min(self.pca_dim, X.shape[0], X.shape[1]))
        if n_comp <= 0:
            n_comp = 1
        self._pca = PCA(n_components=n_comp, random_state=42)
        self._pca.fit(X)
        Z = self._pca.transform(X).astype(np.float32)
        if Z.shape[1] < self.pca_dim:
            pad = np.zeros((Z.shape[0], self.pca_dim - Z.shape[1]), dtype=Z.dtype)
            Z = np.concatenate([Z, pad], axis=1)
        return Z

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        Z = self._pca.transform(X).astype(np.float32)
        if Z.shape[1] < self.pca_dim:
            pad = np.zeros((Z.shape[0], self.pca_dim - Z.shape[1]), dtype=Z.dtype)
            Z = np.concatenate([Z, pad], axis=1)
        return Z

    def build_or_load_cache(self, df_sections: pd.DataFrame, key_cols=("gvkey", "filing_date", "section")) -> pd.DataFrame:
        if self.cache_path.exists():
            return pd.read_parquet(self.cache_path)
        emb = self._encode_chunks(df_sections["text_clean"].tolist(), max_len=512)
        Z = self.fit_transform_pca(emb)
        cols = [f"pca_{i}" for i in range(self.pca_dim)]
        out = pd.concat(
            [df_sections[list(key_cols)].reset_index(drop=True), pd.DataFrame(Z, columns=cols)], axis=1
        )
        out.to_parquet(self.cache_path, index=False)
        return out
