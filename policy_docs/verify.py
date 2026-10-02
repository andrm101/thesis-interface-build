"""Check every extracted quote against the document's own text.

A quote counts as verified when it appears (fuzzy match ≥ THRESHOLD, which
tolerates hyphenation, line breaks and ligatures) on the stated page or a
neighbouring one. Unverified events are kept but flagged, and are not used
as event dates.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

THRESHOLD = 90


def _norm(s: str) -> str:
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("–", "-")
    s = re.sub(r"-\s*\n\s*", "", s)                # hyphenation across lines
    return re.sub(r"\s+", " ", s).strip().lower()


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in ("p", "div", "li", "br", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def page_texts(path: str | Path) -> list[str]:
    """Text per page (PDF) or the whole document as one page (HTML/text)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        return [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suf in (".html", ".htm"):
        p = _Text()
        p.feed(raw)
        raw = "".join(p.parts)
    return [raw]


def verify_quote(quote: str, page: int, pages: list[str]) -> dict:
    from rapidfuzz import fuzz
    q = _norm(quote)
    if len(q) < 12:
        return {"verified": False, "score": 0.0, "found_page": None}
    order = [page - 1, page - 2, page] + [i for i in range(len(pages))
                                          if i not in (page - 1, page - 2, page)]
    best = (0.0, None)
    for i in order:
        if 0 <= i < len(pages):
            sc = fuzz.partial_ratio(q, _norm(pages[i]))
            if sc > best[0]:
                best = (sc, i + 1)
            if sc >= THRESHOLD and abs(i + 1 - page) <= 1:
                break
    score, found = best
    return {"verified": score >= THRESHOLD, "score": round(score, 1),
            "found_page": found}
