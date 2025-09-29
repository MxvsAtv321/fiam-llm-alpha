import os
from fiam_llm.text_clean import clean_html, normalize, sentence_split


def test_text_cleaning_pipeline():
    raw = "<p>Hello&nbsp;WORLD!</p> Visit https://example.com  now."
    c = clean_html(raw)
    assert "<" not in c and "http" not in c
    n = normalize(raw)
    assert n == n.lower()
    assert "  " not in n


def test_sentence_split_deterministic():
    text = "A. B? C!"
    s = sentence_split(text)
    assert s == ["A.", "B?", "C!"]
