"""
extract.py — R&D tax-incentive reforms from policy documents, with quotes.

    python -m policy_docs.extract                 # every new/changed document
    python -m policy_docs.extract --country Poland --force

Input:  data/policy_docs/<Country>/*.pdf | *.html | *.txt
        (e.g. OECD "R&D tax incentives: <country>" profiles, national
        legislation notes). Raw documents stay local (git-ignored).
Output: data/policy_events/extracted/<Country>__<file>.json — the model's
        structured answer plus a quote check per event. These files are the
        frozen record: reproduce.py and the apps read them and never call
        the API, so results stay reproducible.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from llm import client as L
from policy_docs.schema import DocumentExtraction
from policy_docs.verify import page_texts, verify_quote

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "policy_docs"
OUT_DIR = ROOT / "data" / "policy_events" / "extracted"
PROMPT_VERSION = "2026-10-1"
SUFFIXES = (".pdf", ".html", ".htm", ".txt")

SYSTEM = """You extract dated changes to R&D tax incentives from policy documents \
for an economics thesis on EU countries, 2000-2023.

Record an event for every change that altered the generosity or design of an \
R&D tax incentive: a new instrument, a higher or lower rate, a new or removed \
ceiling, a change in eligible firms or costs, or abolition. Ignore direct \
grants and subsidies, and ignore changes that are only proposed.

Rules:
- `quote` must be copied verbatim from the document (one or two sentences), \
so that it can be found by text search. Do not paraphrase or merge sentences.
- `page` is the 1-based page of the document where the quote appears.
- `year_effective` is the year the change applied to firms; use \
`year_announced` for the year it was legislated, if the document gives it.
- Use null for rates the document does not state; never infer them.
- `confidence` is low when the year or the direction is ambiguous.
- If the document describes no such change, return an empty `events` list."""


def _document_block(path: Path) -> dict:
    if path.suffix.lower() == ".pdf":
        data = base64.standard_b64encode(path.read_bytes()).decode()
        src = {"type": "base64", "media_type": "application/pdf", "data": data}
    else:
        text = "\n".join(page_texts(path))
        src = {"type": "text", "media_type": "text/plain", "data": text}
    return {"type": "document", "source": src, "title": path.name}


def out_path(country: str, path: Path) -> Path:
    return OUT_DIR / f"{country}__{path.stem}.json"


def extract_document(path: Path, country: str, client=None) -> dict:
    """Run the extraction for one document and check every quote."""
    client = client or L.get_client()
    resp = L.parse(
        client,
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            _document_block(path),
            {"type": "text", "text": f"Country: {country}. Extract the R&D "
                                     "tax-incentive reform events."}]}],
        output_format=DocumentExtraction,
    )
    ext: DocumentExtraction = resp.parsed_output
    pages = page_texts(path)
    checks = [verify_quote(e.quote, e.page, pages) for e in ext.events]
    return {
        "file": f"{country}/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": getattr(resp, "model", L.MODEL),
        "prompt_version": PROMPT_VERSION,
        "country": country,
        "extraction": ext.model_dump(),
        "checks": checks,
    }


def _up_to_date(out: Path, sha: str) -> bool:
    if not out.exists():
        return False
    rec = json.loads(out.read_text(encoding="utf-8"))
    return rec.get("sha256") == sha and rec.get("prompt_version") == PROMPT_VERSION


def documents(doc_dir: Path = DOC_DIR, country: str | None = None):
    if not doc_dir.is_dir():
        return
    for cdir in sorted(p for p in doc_dir.iterdir() if p.is_dir()):
        if country and cdir.name != country:
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in SUFFIXES:
                yield cdir.name, f


def run(doc_dir: Path = DOC_DIR, out_dir: Path = OUT_DIR, country=None,
        force=False, client=None, log=print) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c, f in documents(doc_dir, country):
        out = out_dir / f"{c}__{f.stem}.json"
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if not force and _up_to_date(out, sha):
            log(f"  {c}/{f.name}: up to date")
            continue
        rec = extract_document(f, c, client=client)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        ok = sum(ch["verified"] for ch in rec["checks"])
        log(f"  {c}/{f.name}: {len(rec['checks'])} events, {ok} quotes verified")
        written.append(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DOC_DIR)
    ap.add_argument("--country")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the document is unchanged")
    a = ap.parse_args(argv)
    n = run(a.dir, OUT_DIR, a.country, a.force)
    print(f"{len(n)} document(s) extracted → {OUT_DIR.relative_to(ROOT)}")
    from policy_docs import reconcile
    reconcile.main([])


if __name__ == "__main__":
    main()
