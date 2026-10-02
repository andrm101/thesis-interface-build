"""policy_docs/: structured extraction (fake Claude client), quote checks,
caching, reconciliation with the derived dates, and the LLM client guard."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("pypdf")
pytest.importorskip("rapidfuzz")

from llm import client as L
from policy_docs import extract, reconcile
from policy_docs.schema import DocumentExtraction, ReformEvent
from policy_docs.verify import page_texts, verify_quote

TRUE_QUOTE = "In 2016 Poland introduced an enhanced deduction of 150% of eligible R&D costs."


def _pdf(path, pages):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for txt in pages:
            f = plt.figure(figsize=(8.5, 11))
            f.text(0.08, 0.8, txt, fontsize=11)
            pdf.savefig(f)
            plt.close(f)


def _event(**kw):
    base = {"year_effective": 2016, "year_announced": 2015, "instrument": "enhanced_deduction",
                "direction": "introduction", "firm_scope": "all", "rate_before": None,
                "rate_after": 1.5, "summary": "New R&D super-deduction.", "quote": TRUE_QUOTE,
                "page": 2, "confidence": "high"}
    base.update(kw)
    return ReformEvent(**base)


class FakeClient:
    """Stands in for anthropic.Anthropic: records calls, returns a parsed answer."""
    def __init__(self, parsed, stop_reason="end_turn"):
        self.calls = []
        outer = self

        class _Msgs:
            def parse(self, **kw):
                outer.calls.append(kw)
                return SimpleNamespace(parsed_output=parsed, model="claude-opus-5",
                                       stop_reason=stop_reason, stop_details=None)
        self.beta = SimpleNamespace(messages=_Msgs())


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "policy_docs" / "Poland"
    d.mkdir(parents=True)
    _pdf(d / "profile.pdf", ["OECD R&D tax incentives: Poland.",
                             ("In 2016 Poland introduced an enhanced\ndeduction of "
                              "150% of eligible R&D costs.")])
    return tmp_path


def test_verify_quote_finds_page_and_rejects_invented(docs):
    pages = page_texts(docs / "policy_docs" / "Poland" / "profile.pdf")
    assert verify_quote(TRUE_QUOTE, 2, pages) == {"verified": True, "score": 100.0, "found_page": 2}
    assert verify_quote(TRUE_QUOTE, 1, pages)["verified"]          # off-by-one page tolerated
    assert not verify_quote("Poland abolished its patent box regime in 2016.", 2, pages)["verified"]


def test_extract_run_checks_quotes_and_caches(docs):
    parsed = DocumentExtraction(country="Poland", source_title="OECD profile", events=[
        _event(), _event(year_effective=2018, direction="expansion",
                         quote="Poland raised the deduction to 200% in 2018.")])
    fake = FakeClient(parsed)
    out = docs / "extracted"
    written = extract.run(docs / "policy_docs", out, client=fake, log=lambda *_: None)
    assert len(written) == 1 and len(fake.calls) == 1
    call = fake.calls[0]
    assert call["output_format"] is DocumentExtraction
    assert call["fallbacks"] == "default" and L.FALLBACK_BETA in call["betas"]
    doc_block = call["messages"][0]["content"][0]
    assert doc_block["source"]["media_type"] == "application/pdf"
    rec = json.loads(written[0].read_text())
    assert [c["verified"] for c in rec["checks"]] == [True, False]   # 2nd quote is not in the PDF
    # unchanged document → no second API call
    assert extract.run(docs / "policy_docs", out, client=fake, log=lambda *_: None) == []
    assert len(fake.calls) == 1


def test_refusal_raises():
    with pytest.raises(L.LLMRefusal):
        L.check(SimpleNamespace(stop_reason="refusal",
                                stop_details=SimpleNamespace(category="cyber")))


def test_availability_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    ok, why = L.availability()
    assert not ok and "ANTHROPIC_API_KEY" in why
    with pytest.raises(L.LLMUnavailable):
        L.get_client()


def test_reconcile_statuses():
    ev = pd.DataFrame([
        {"Country": "Poland", "year_effective": 2016, "direction": "introduction",
             "confidence": "high", "verified": True, "instrument": "enhanced_deduction",
             "source": "Poland/p.pdf", "quote": "q"},
        {"Country": "Poland", "year_effective": 2018, "direction": "expansion",
             "confidence": "high", "verified": True, "instrument": "enhanced_deduction",
             "source": "Poland/p.pdf", "quote": "q"},
        {"Country": "Czechia", "year_effective": 2008, "direction": "introduction",
             "confidence": "high", "verified": True, "instrument": "enhanced_deduction",
             "source": "Czechia/p.pdf", "quote": "q"},
        {"Country": "Latvia", "year_effective": 2014, "direction": "introduction",
             "confidence": "high", "verified": False, "instrument": "other",
             "source": "Latvia/p.pdf", "quote": "q"},           # unverified → ignored
        {"Country": "Greece", "year_effective": 2013, "direction": "restriction",
             "confidence": "high", "verified": True, "instrument": "other",
             "source": "Greece/p.pdf", "quote": "q"},           # not a generosity rise
    ])
    doc = reconcile.documented_events(ev)
    assert dict(zip(doc.Country, doc.Year)) == {"Poland": 2016, "Czechia": 2008}
    rec = reconcile.reconcile(doc, {"Poland": 2016, "Czechia": 2005, "Sweden": 2014})
    st = dict(zip(rec.Country, rec.status))
    assert st == {"Czechia": "date_shift", "Poland": "confirmed", "Sweden": "derived_only"}
