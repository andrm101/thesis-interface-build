"""
service.py — the analysis layer behind the web API.

Loads the rebuilt level panel once (with the Tab 1 default z-screen, exactly
as reproduce.py Track B does) and exposes JSON-ready wrappers around the
existing analysis modules. No analysis logic lives here: every number comes
from causal.py, clubs.py, outliers.py or reproduce.py.
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import causal
import clubs
import reproduce as R

ROOT = R.HERE
RESULTS_MD = os.path.join(ROOT, "results", "RESULTS.md")
FIG_DIR = os.path.join(ROOT, "results", "figures")
WINDOW = (2000, 2023)

# Human-readable labels and units for the variables the web app offers.
VARIABLES: dict[str, tuple[str, str]] = {
    "Y_per_worker": ("Output per worker", "2015 USD"),
    "GDP_pc_2015usd": ("GDP per capita", "2015 USD"),
    "Frontier_gap": ("Distance to productivity frontier", "log points"),
    "RD_pct_GDP": ("R&D intensity", "% of GDP"),
    "Researchers_per_1000_emp": ("Researchers", "per 1,000 employed"),
    "Patents_per_million": ("Patent applications", "per million people"),
    "Tertiary_share": ("Tertiary education share", "share"),
    "RD_stock_per_worker": ("R&D capital stock per worker", "2015 USD"),
    "Savings_rate": ("Gross savings", "% of GDP"),
    "Employment": ("Employment", "persons"),
    "GBARD_pct_GDP": ("Public R&D budget (GBARD)", "% of GDP"),
    "NonGBARD_RD_pct_GDP": ("R&D not financed by the budget", "% of GDP"),
    "GBARD_share_GERD": ("GBARD relative to total R&D", "ratio"),
    "RD_subsidy_large_profit": ("R&D tax subsidy, large profitable firm",
                                "1 − B-index"),
    "RD_subsidy_sme_profit": ("R&D tax subsidy, profitable SME",
                              "1 − B-index"),
    "Gov_balance_pct_GDP": ("Government balance (headline)", "% of GDP"),
    "CAB_pct_potGDP": ("Government balance (cyclically adjusted)",
                       "% of potential GDP"),
}
# Shocks that make sense for local projections (one unit = one pp etc.).
SHOCKS = ["RD_pct_GDP", "GBARD_pct_GDP", "RD_subsidy_large_profit",
          "Gov_balance_pct_GDP", "CAB_pct_potGDP", "Savings_rate"]


def _f(x):
    """JSON-safe float (NaN/inf → None)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


@lru_cache(maxsize=1)
def panel() -> tuple[pd.DataFrame, str]:
    _, scr = R.load_screened(R.LEVELS)
    return scr.df.copy(), scr.summary()


def df() -> pd.DataFrame:
    return panel()[0]


def groups() -> dict[str, str]:
    return R.groups_apriori(df()).to_dict()


def variables() -> list[dict]:
    d = df()
    out = []
    for key, (label, unit) in VARIABLES.items():
        if key not in d:
            continue
        sub = d[d["Year"].between(*WINDOW)]
        n = int(sub[key].notna().sum())
        out.append({"key": key, "label": label, "unit": unit,
                    "coverage": round(n / max(len(sub), 1), 3),
                    "shock": key in SHOCKS})
    return out


def event_sets() -> dict[str, dict[str, int]]:
    d = df()
    sets = {"EU accession": {c: y for c, y in causal.EU_ACCESSION.items()
                             if c in set(d["Country"])}}
    if "RD_tax_reform_year" in d:
        ev = (d.dropna(subset=["RD_tax_reform_year"]).groupby("Country")
              ["RD_tax_reform_year"].first().astype(int))
        sets["R&D tax reforms"] = ev.to_dict()
    from policy_docs.reconcile import load_documented
    doc = {c: y for c, y in load_documented().items() if c in set(d["Country"])}
    if doc:
        sets["R&D tax reforms (documented)"] = doc
    return sets


def meta() -> dict:
    d, summary = panel()
    g = groups()
    return {
        "countries": [{"name": c, "group": g[c]}
                      for c in sorted(d["Country"].unique())],
        "years": [int(d["Year"].min()), int(d["Year"].max())],
        "variables": variables(),
        "event_sets": event_sets(),
        "control_groups": list(causal.CONTROL_GROUPS),
        "screen": summary,
    }


def _check_var(v: str):
    if v not in VARIABLES or v not in df():
        raise KeyError(f"unknown variable: {v}")


def series(variable: str, countries: list[str] | None = None) -> dict:
    _check_var(variable)
    d = df()
    years = sorted(int(y) for y in d["Year"].unique())
    g = groups()
    w = d.pivot_table(index="Year", columns="Country", values=variable)
    w = w.reindex(years)
    names = countries or list(w.columns)
    out = [{"country": c, "group": g.get(c),
            "values": [_f(v) for v in w[c]]} for c in names if c in w]
    return {"variable": variable, "years": years, "series": out}


def snapshot(variable: str, year: int) -> dict:
    _check_var(variable)
    d = df()
    s = d[d["Year"] == year].set_index("Country")[variable].dropna()
    g = groups()
    rows = [{"country": c, "group": g.get(c), "value": _f(v)}
            for c, v in s.sort_values(ascending=False).items()]
    return {"variable": variable, "year": year, "rows": rows}


@lru_cache(maxsize=1)
def typology() -> dict:
    d = df()
    km = R.kmeans_typology(d, R.LEVEL_RD)
    cm = d.groupby("Country")[R.LEVEL_RD].mean().dropna()
    X = StandardScaler().fit_transform(cm)
    pca = PCA(2, random_state=0).fit(X)
    xy = pca.transform(X)
    inno = set(km["innovative"])
    g = groups()
    pts = [{"country": c, "x": _f(x), "y": _f(y),
            "cluster": "Innovative" if c in inno else "Emerging",
            "group": g.get(c)} for c, (x, y) in zip(cm.index, xy)]
    load = [{"variable": v, "pc1": _f(a), "pc2": _f(b)}
            for v, a, b in zip(R.LEVEL_RD, *pca.components_)]
    return {"points": pts, "loadings": load,
            "explained": [_f(e) for e in pca.explained_variance_ratio_],
            "silhouette": _f(km["silhouette"]),
            "agreement": _f(km["agreement"])}


@lru_cache(maxsize=4)
def convergence_clubs(variable: str = "Y_per_worker") -> dict:
    _check_var(variable)
    lx = clubs.prepare_panel(df(), variable, chain=False)
    res = clubs.club_clustering(lx)
    h = clubs.transition_paths(lx)
    mem = res.membership().set_index("Country")["Club"].to_dict()
    return {
        "variable": variable,
        "years": [int(y) for y in h.index],
        "full_t": _f(res.full.t), "converges": bool(res.full.converges),
        "clubs": [{"club": k + 1, "t": _f(t.t), "members": m}
                  for k, (m, t) in enumerate(zip(res.clubs, res.club_tests))],
        "divergent": res.divergent,
        "paths": [{"country": c, "club": int(mem.get(c, 0)),
                   "values": [_f(v) for v in h[c]]} for c in h.columns],
    }


def local_projection(outcome: str, shock: str, horizons: int = 5,
                     lags: int = 1, levels: bool = False,
                     split: bool = False) -> dict:
    _check_var(outcome)
    _check_var(shock)
    d = df()
    state = None
    if split:
        state = (d["Country"].map(groups()) == "Emerging").astype(float)
    lp = causal.local_projections(
        d, outcome, shock, horizons, lags, state=state,
        state_names=("Emerging", "Innovative"), levels=levels)
    rows = [{"h": int(r.h), "state": r.state, "beta": _f(r.beta),
             "se": _f(r.se), "lo": _f(r.lo), "hi": _f(r.hi), "p": _f(r.p),
             "n": int(r.n)} for r in lp.itertuples()]
    unit = (VARIABLES[outcome][1] if levels else "% change")
    return {"outcome": outcome, "shock": shock, "levels": levels,
            "response_unit": unit, "shock_unit": VARIABLES[shock][1],
            "rows": rows}


def event_study(outcome: str, event_set: str, custom: str = "",
                control: str = causal.CONTROL_GROUPS[1],
                detrend: bool = False, pre: int = 3, post: int = 5,
                n_boot: int = 99) -> dict:
    _check_var(outcome)
    if control not in causal.CONTROL_GROUPS:
        raise KeyError(f"unknown control group: {control}")
    if event_set == "Custom":
        ev = causal.parse_events(custom)
    else:
        sets = event_sets()
        if event_set not in sets:
            raise KeyError(f"unknown event set: {event_set}")
        ev = sets[event_set]
    ev = {c: y for c, y in ev.items() if c in set(df()["Country"])}
    if not ev:
        raise ValueError("no valid events (format: Country:Year, …)")
    r = causal.event_study(df(), outcome, ev, pre, post,
                           min(int(n_boot), 499), control=control,
                           detrend=detrend)
    et = [{"e": int(x.e), "att": _f(x.att), "se": _f(x.se), "lo": _f(x.lo),
           "hi": _f(x.hi), "n_cohorts": int(x.n_cohorts)}
          for x in r.by_event_time.itertuples()]
    return {"outcome": outcome, "events": ev, "by_event_time": et,
            "overall_post": _f(r.overall_post),
            "overall_post_se": _f(r.overall_post_se),
            "pretrend_p": _f(r.pretrend_p), "treated": r.treated,
            "controls": r.controls}


def coverage() -> dict:
    d = df()
    sub = d[d["Year"].between(*WINDOW)]
    n_years = WINDOW[1] - WINDOW[0] + 1
    keys = [v["key"] for v in variables()]
    cells = []
    for c, g in sub.groupby("Country"):
        for k in keys:
            cells.append({"country": c, "variable": k,
                          "share": _f(g[k].notna().sum() / n_years)})
    return {"window": list(WINDOW), "variables": keys,
            "countries": sorted(sub["Country"].unique()), "cells": cells}


# ── results/RESULTS.md and figures ─────────────────────────────────────────
_SEC = re.compile(r"^\*\*([A-D]\d+b?) · (.+?)(?:\*\*|$)")
TRACKS = {"A": "Thesis panel", "B": "Rebuilt panel", "C": "Convergence",
          "D": "Policy variables"}
FIG_TITLES = {
    "fig1_typology_pca": "K-Means typology on R&D levels",
    "fig2_convergence_clubs": "Phillips–Sul convergence clubs",
    "fig3_jcurve_lags": "R&D effect by lag and group",
    "fig4_event_study": "EU-accession event study",
    "fig5_gmm_specifications": "System GMM across specifications",
    "fig6_dml_sensitivity": "Double ML sensitivity",
    "fig7_beta_convergence": "β-convergence by typology",
    "fig8_tax_reforms": "R&D tax-incentive reforms",
}


@lru_cache(maxsize=1)
def _sections() -> list[dict]:
    if not os.path.exists(RESULTS_MD):
        return []
    with open(RESULTS_MD, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out, cur = [], None
    for ln in lines:
        if ln.startswith("## "):
            if cur:
                out.append(cur)
                cur = None
            continue
        m = _SEC.match(ln)
        if m:
            if cur:
                out.append(cur)
            title = re.sub(r"[*`]", "", m.group(2)).rstrip(":. ")
            cur = {"id": m.group(1), "title": title,
                   "track": TRACKS[m.group(1)[0]],
                   "lines": [ln]}
        elif cur:
            cur["lines"].append(ln)
    if cur:
        out.append(cur)
    return out


def results_index() -> list[dict]:
    return [{k: s[k] for k in ("id", "title", "track")} for s in _sections()]


def result(section_id: str) -> dict:
    for s in _sections():
        if s["id"].lower() == section_id.lower():
            return {"id": s["id"], "title": s["title"], "track": s["track"],
                    "markdown": "\n".join(s["lines"]).strip()}
    raise KeyError(f"unknown section: {section_id}")


def figures() -> list[dict]:
    if not os.path.isdir(FIG_DIR):
        return []
    out = []
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith(".png"):
            stem = f[:-4]
            title = FIG_TITLES.get(
                stem, re.sub(r"^fig\d+_", "", stem).replace("_", " "))
            out.append({"name": stem, "title": title,
                        "png": f"/figures/{f}",
                        "pdf": f"/figures/{stem}.pdf"})
    return out


def warm() -> None:
    """Pre-compute the cached views (called at start-up)."""
    panel()
    typology()
    convergence_clubs()
    np.seterr(all="ignore")
