"""
reconcile.py — documented reform dates vs the dates derived from the OECD
implied-subsidy series (policy_data.tax_reform_events).

    python -m policy_docs.reconcile

Writes data/policy_events/documented_events.csv (one generosity-raising
reform per country: the earliest verified introduction or expansion in the
event-study window) and data/policy_events/reconciliation.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "policy_events" / "extracted"
DOCUMENTED = ROOT / "data" / "policy_events" / "documented_events.csv"
RECON = ROOT / "data" / "policy_events" / "reconciliation.csv"
WINDOW = (2002, 2021)          # same first year as the derived dates; ≥ 2 post years


def load_events(extracted: Path = EXTRACTED) -> pd.DataFrame:
    rows = []
    for f in sorted(extracted.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        for ev, ch in zip(rec["extraction"]["events"], rec["checks"]):
            rows.append({"Country": rec["country"], "source": rec["file"],
                         **{k: ev[k] for k in ("year_effective", "year_announced",
                                               "instrument", "direction",
                                               "firm_scope", "confidence",
                                               "summary", "quote", "page")},
                         "verified": ch["verified"], "match_score": ch["score"]})
    return pd.DataFrame(rows)


def documented_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Earliest verified, generosity-raising reform per country."""
    if ev.empty:
        return pd.DataFrame(columns=["Country", "Year", "instrument", "source", "quote"])
    ok = ev[ev["verified"] & ev["direction"].isin(["introduction", "expansion"])
            & ev["confidence"].isin(["medium", "high"])
            & ev["year_effective"].between(*WINDOW)]
    first = ok.sort_values("year_effective").groupby("Country").first().reset_index()
    return first.rename(columns={"year_effective": "Year"})[
        ["Country", "Year", "instrument", "source", "quote"]]


def reconcile(documented: pd.DataFrame, derived: dict[str, int]) -> pd.DataFrame:
    doc = dict(zip(documented["Country"], documented["Year"].astype(int)))
    rows = []
    for c in sorted(set(doc) | set(derived)):
        d, g = doc.get(c), derived.get(c)
        if d is not None and g is not None:
            st = "confirmed" if abs(d - g) <= 1 else "date_shift"
        else:
            st = "doc_only" if d is not None else "derived_only"
        rows.append({"Country": c, "derived_year": g, "documented_year": d,
                     "gap_years": None if d is None or g is None else d - g,
                     "status": st, "reviewed": ""})
    return pd.DataFrame(rows)


def load_documented(path: Path = DOCUMENTED) -> dict[str, int]:
    """{country: year} for the event study; {} when nothing is extracted."""
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    return {c: int(y) for c, y in zip(d["Country"], d["Year"])}


def main(argv=None):
    ev = load_events()
    if ev.empty:
        print("no extracted documents yet (run python -m policy_docs.extract)")
        return
    doc = documented_events(ev)
    DOCUMENTED.parent.mkdir(parents=True, exist_ok=True)
    doc.to_csv(DOCUMENTED, index=False)
    panel = pd.read_csv(ROOT / "data" / "panel_levels.csv")
    derived = (panel.dropna(subset=["RD_tax_reform_year"])
               .groupby("Country")["RD_tax_reform_year"].first().astype(int)
               .to_dict()) if "RD_tax_reform_year" in panel else {}
    rec = reconcile(doc, derived)
    rec.to_csv(RECON, index=False)
    counts = rec["status"].value_counts().to_dict()
    print(f"{len(doc)} documented reform dates; reconciliation: {counts}")


if __name__ == "__main__":
    main()
