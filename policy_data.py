"""
policy_data.py — Import policy variables (R&D budgets, R&D tax incentives,
fiscal balance) from Eurostat / OECD spreadsheet exports, splice them over
1998-2023, report coverage, and derive policy-event dates.

Expected files in data/raw/policy/ (any Eurostat "spreadsheet" export or
OECD Data Explorer table export with the default layout):

  gba_nabsfin92.xlsx   GBARD, NABS 1992 (years up to 2007)  — sheet with
                       "Total appropriations" in "Million euro"
  gba_nabsfin07.xlsx   GBARD, NABS 2007 (2007 onwards)      — "Total GBARD"
  gov_10dd_edpt1.xlsx  General government net lending, % of GDP
  oecd_rdsub.xlsx      OECD implied R&D tax subsidy rates (1 − B-index),
                       SME / large firm × profitable / loss-making

Splice rule for GBARD: only the *breakdown by objective* changed between
NABS 1992 and NABS 2007, not the total, so the totals are spliced
directly: NABS 2007 values where present, NABS 1992 otherwise. Every value
carries its source in the `_src` column; `coverage()` reports gaps.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "data", "raw", "policy")
FILES = {"gba92": "gba_nabsfin92.xlsx", "gba07": "gba_nabsfin07.xlsx",
         "gov": "gov_10dd_edpt1.xlsx", "rdsub": "oecd_rdsub.xlsx"}
WINDOW = (1998, 2023)

NAME_FIX = {"Slovak Republic": "Slovakia", "Czech Republic": "Czechia",
            "Türkiye": "Turkey"}


def _clean_name(s: str) -> str:
    s = re.sub(r"^[·\s ]+", "", str(s)).strip()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)          # "Germany (until 1990 …)"
    return NAME_FIX.get(s, s)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _year(v):
    m = re.match(r"^\s*(\d{4})", str(v))
    return int(m.group(1)) if m else None


# ── Eurostat spreadsheet exports ───────────────────────────────────────────
def _eurostat_meta(d: pd.DataFrame):
    meta, time_row = {}, None
    for i in range(min(25, len(d))):
        k = str(d.iat[i, 0]).strip()
        if k == "TIME":
            time_row = i
            break
        if d.shape[1] > 2 and str(d.iat[i, 2]) != "nan":
            meta[k] = str(d.iat[i, 2]).strip()
    return meta, time_row


def read_eurostat(path: str, **criteria) -> pd.DataFrame:
    """Long frame (Country, Year, value, flag) from a Eurostat spreadsheet
    export. With several sheets, pick the one whose metadata values contain
    every criteria string (e.g. unit="Million euro", nomenclature="Total")."""
    xl = pd.ExcelFile(path, engine="calamine")
    chosen = None
    for sh in xl.sheet_names:
        d = pd.read_excel(xl, sheet_name=sh, header=None)
        meta, t = _eurostat_meta(d)
        if t is None:
            continue
        ok = True
        for key, want in criteria.items():
            vals = [v for k, v in meta.items() if k.lower().startswith(key)]
            if not any(v.lower() == want.lower() or v.lower().startswith(want.lower())
                       for v in vals):
                ok = False
        if ok:
            chosen = (sh, d, meta, t)
            break
    if chosen is None:
        raise ValueError(f"{os.path.basename(path)}: no sheet matches {criteria}")
    sh, d, meta, t = chosen
    years = {j: _year(v) for j, v in enumerate(d.iloc[t]) if _year(v)}
    rows = []
    for i in range(t + 1, len(d)):
        name = str(d.iat[i, 0]).strip()
        if name in ("nan", "GEO (Labels)"):
            continue
        if name.startswith(("Special value", "Observation flags", ":")):
            break
        for j, y in years.items():
            flag = str(d.iat[i, j + 1]).strip() if j + 1 < d.shape[1] else ""
            rows.append({"Country": _clean_name(name), "Year": y,
                         "value": _num(d.iat[i, j]),
                         "flag": "" if flag == "nan" else flag})
    out = pd.DataFrame(rows)
    out.attrs.update(sheet=sh, **{k: v for k, v in meta.items()})
    return out


# ── OECD Data Explorer table export ────────────────────────────────────────
def read_oecd_rdsub(path: str) -> pd.DataFrame:
    """Long frame (Country, Year, firm, scenario, value) from the OECD
    'Implied tax subsidy rates on R&D expenditures' table export."""
    d = pd.read_excel(path, sheet_name=0, engine="calamine", header=None)
    hdr = next(i for i in range(15)
               if any(_year(v) for v in d.iloc[i].tolist()[3:]))
    years = {j: _year(v) for j, v in enumerate(d.iloc[hdr]) if _year(v)}
    rows, scenario = [], None
    for i in range(hdr + 1, len(d)):
        a, b = str(d.iat[i, 1]).strip(), str(d.iat[i, 2]).strip()
        if a.startswith("Profit scenario"):
            scenario = a.split(":", 1)[1].strip().lower()
            continue
        if b in ("SME", "Large firm") and scenario:
            for j, y in years.items():
                rows.append({"Country": _clean_name(a), "Year": y,
                             "firm": "sme" if b == "SME" else "large",
                             "scenario": "profit" if scenario.startswith("profit")
                             else "loss", "value": _num(d.iat[i, j])})
    return pd.DataFrame(rows)


# ── assemble ───────────────────────────────────────────────────────────────
def build(policy_dir: str = POLICY_DIR) -> pd.DataFrame:
    """Country × Year frame of policy variables (+ `_src` provenance)."""
    p = lambda k: os.path.join(policy_dir, FILES[k])  # noqa: E731
    parts = []
    if os.path.exists(p("gba92")) or os.path.exists(p("gba07")):
        g = []
        if os.path.exists(p("gba92")):
            g92 = read_eurostat(p("gba92"), **{"unit of measure": "Million euro",
                                               "nomenclature": "Total appropriations"})
            g.append(g92.assign(src="NABS1992"))
        if os.path.exists(p("gba07")):
            g07 = read_eurostat(p("gba07"), **{"unit of measure": "Million euro",
                                               "nomenclature": "Total"})
            g.append(g07.assign(src="NABS2007"))
        g = pd.concat(g).dropna(subset=["value"])
        g["rank"] = (g["src"] == "NABS2007").astype(int)
        g = (g.sort_values("rank").groupby(["Country", "Year"]).last()
             .reset_index())
        parts.append(g.rename(columns={"value": "GBARD_mEUR",
                                       "src": "GBARD_src"})
                     [["Country", "Year", "GBARD_mEUR", "GBARD_src"]])
    if os.path.exists(p("gov")):
        v = read_eurostat(p("gov"), **{"unit of measure": "Percentage of gross"})
        parts.append(v.rename(columns={"value": "Gov_balance_pct_GDP"})
                     [["Country", "Year", "Gov_balance_pct_GDP"]])
    if os.path.exists(p("rdsub")):
        s = read_oecd_rdsub(p("rdsub"))
        s["col"] = "RD_subsidy_" + s["firm"] + "_" + s["scenario"]
        w = s.pivot_table(index=["Country", "Year"], columns="col",
                          values="value").reset_index()
        w.columns.name = None
        parts.append(w)
    if not parts:
        return pd.DataFrame(columns=["Country", "Year"])
    out = parts[0]
    for q in parts[1:]:
        out = out.merge(q, on=["Country", "Year"], how="outer")
    return out.sort_values(["Country", "Year"]).reset_index(drop=True)


def coverage(df: pd.DataFrame, countries, cols=None,
             window=WINDOW) -> pd.DataFrame:
    """Country × variable coverage inside *window*: 'first–last (n missing)'."""
    cols = cols or [c for c in df.columns
                    if c not in ("Country", "Year") and not c.endswith("_src")]
    yrs = range(window[0], window[1] + 1)
    rows = []
    for c in countries:
        rec = {"Country": c}
        sub = df[df["Country"] == c].set_index("Year")
        for v in cols:
            s = sub[v].reindex(yrs) if v in sub else pd.Series(index=yrs,
                                                                 dtype=float)
            have = s.dropna()
            rec[v] = ("—" if have.empty else
                      f"{have.index.min()}–{have.index.max()}"
                      + (f" ({s.isna().sum()} gaps)" if s.isna().sum() else ""))
        rows.append(rec)
    return pd.DataFrame(rows)


# ── policy events ──────────────────────────────────────────────────────────
def tax_reform_events(df: pd.DataFrame, col="RD_subsidy_large_profit",
                      threshold: float = 0.05, first_year: int = 2002,
                      persist: int = 2) -> dict[str, int]:
    """First year a country's implied R&D tax subsidy rises by ≥ threshold
    (e.g. 5 pp) and stays at least that much above its pre-reform level for
    *persist* further years — a generosity reform. Countries without such a
    jump are never-treated."""
    ev = {}
    for c, g in df.sort_values("Year").groupby("Country"):
        s = g.set_index("Year")[col].dropna()
        for y in s.index:
            if y < first_year or (y - 1) not in s.index:
                continue
            base = s[y - 1]
            later = s.loc[y:y + persist]
            if s[y] - base >= threshold and (later - base >= threshold).all() \
                    and len(later) == persist + 1:
                ev[c] = int(y)
                break
    return ev


def consolidation_episodes(df: pd.DataFrame, col="Gov_balance_pct_GDP",
                           threshold: float = 1.5) -> pd.Series:
    """0/1 indicator: government balance improves by ≥ threshold pp of GDP
    in a year (a large fiscal tightening; headline balance, so cyclical
    swings are not removed — see docs)."""
    d = df.sort_values(["Country", "Year"])
    dbal = d.groupby("Country")[col].diff()
    return (dbal >= threshold).astype(float).where(dbal.notna()).reindex(df.index)
