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

    def _tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        return tokens[: self.cfg.max_tokens]

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        # Deterministic bag-of-words hashing to 256 dims as stub feature
        vec = np.zeros(256, dtype=np.float32)
        for t in tokens:
            h = abs(hash(t)) % 256
            vec[h] += 1.0
        if vec.sum() > 0:
            vec = vec / np.linalg.norm(vec)
        return vec

    def fit_pca(self, texts: Iterable[str]) -> None:
        X = np.stack([self.embed_text(t) for t in texts])
        self.pca = PCA(n_components=self.cfg.pca_dim, random_state=self.cfg.seed)
        self.pca.fit(X)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        X = np.stack([self.embed_text(t) for t in texts])
        if self.pca is None:
            return X[:, : self.cfg.pca_dim]
        Z = self.pca.transform(X)
        return Z.astype(np.float32)
