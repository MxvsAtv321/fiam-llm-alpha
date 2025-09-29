import numpy as np
import pandas as pd
import types
import builtins
import importlib
import pathlib
import fiam_llm.embedder as E


class DummyTok:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, text, truncation=False, add_special_tokens=False):
        # Return deterministic token ids by splitting words
        ids = list(range(1, len(text.split()) + 1))
        return {"input_ids": ids}

    def encode_plus(self, ids, is_split_into_words=False, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512):
        L = min(len(ids) + 2, max_length)
        # Fake attention mask of ones
        return {"attention_mask": DummyTensor(np.ones((1, L), dtype=np.float32))}


class DummyOut:
    def __init__(self, L, H):
        self.last_hidden_state = DummyTensor(np.ones((1, L, H), dtype=np.float32))


class DummyModel:
    config = types.SimpleNamespace(hidden_size=8)

    def eval(self):
        return self

    def __call__(self, **enc):
        L = enc["attention_mask"].array.shape[1]
        return DummyOut(L, self.config.hidden_size)


class DummyTensor:
    def __init__(self, array):
        self.array = array

    def float(self):
        return self

    def sum(self, dim=1, keepdim=True):
        s = self.array.sum(axis=dim, keepdims=keepdim)
        return DummyTensor(s)

    def __truediv__(self, other):
        out = self.array / (other.array + 1e-8)
        return DummyTensor(out)

    def unsqueeze(self, dim):
        return self

    def __mul__(self, other):
        return DummyTensor(self.array * other.array)

    def __add__(self, other):
        return DummyTensor(self.array + other.array)

    def sum_(self, axis=None, keepdims=False):
        return DummyTensor(self.array.sum(axis=axis, keepdims=keepdims))

    def sum(self, dim=None, keepdim=False):
        axis = dim
        return DummyTensor(self.array.sum(axis=axis, keepdims=keepdim))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def test_section_embedder_cache_write_and_read(tmp_path, monkeypatch):
    # Monkeypatch HF objects in module
    monkeypatch.setattr(E, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok()))
    monkeypatch.setattr(E, "AutoModel", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()))
    monkeypatch.setattr(E, "torch", None)

    df = pd.DataFrame({
        "gvkey": ["g1", "g2"],
        "filing_date": ["2014-12-15", "2015-01-10"],
        "section": ["MDNA", "RISK"],
        "text_clean": ["profit growth", "risk loss"],
    })
    cache = tmp_path / "sec.parquet"
    se = E.SectionEmbedder(model_name="dummy", pca_dim=4, cache_path=str(cache))
    out1 = se.build_or_load_cache(df)
    assert cache.exists()
    assert all(col.startswith("pca_") for col in out1.columns if col.startswith("pca_"))
    # Types float32
    pca_cols = [c for c in out1.columns if c.startswith("pca_")]
    assert out1[pca_cols].values.dtype == np.float32
    # Second call should load without re-encoding
    out2 = se.build_or_load_cache(df)
    assert out2.shape == out1.shape
