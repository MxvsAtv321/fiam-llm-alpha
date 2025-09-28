from __future__ import annotations

import re
from typing import List

HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_html(text: str) -> str:
    text = HTML_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"http[s]?://\S+", " ", text)
    return WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    return clean_html(text).lower()


def sentence_split(text: str) -> List[str]:
    # Lightweight deterministic splitter
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
