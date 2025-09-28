import numpy as np
from fiam_llm.embedder import DeterministicStubEmbedder, EmbedConfig


def test_embedder_determinism_and_pca_dim():
    cfg = EmbedConfig(model_name="stub", max_tokens=16, pca_dim=8)
    emb = DeterministicStubEmbedder(cfg)
    texts = ["a b c d", "a b c d e f"]
    emb.fit_pca(texts)
    Z1 = emb.transform(texts)
    Z2 = emb.transform(texts)
    assert Z1.shape[1] == cfg.pca_dim
    assert np.allclose(Z1, Z2)
