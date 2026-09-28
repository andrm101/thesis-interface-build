"""
reproduce.py — Regenerate every headline result, headless, in one run.

    python reproduce.py                  # writes results/RESULTS.md
    python reproduce.py --quick          # fewer bootstrap/simulation draws

Two tracks, both run through the same modules the GUI uses:

  Track A  panel_data.xlsx       — the thesis panel as submitted
                                   (growth indices; reproduces the thesis)
  Track B  data/panel_levels.csv — rebuilt from the raw workbook with the
                                   data-audit corrections and level
                                   variables (build_panel.py)

Every stochastic step is seeded, so numbers are identical run to run.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import causal
import clubs
import gmm
import ml_eval
import outliers
import panel_tests
import policy_data
from constants import INNOVATIVE_CLUSTER, stars

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.join(HERE, "panel_data.xlsx")
LEVELS = os.path.join(HERE, "data", "panel_levels.csv")
OUT = os.path.join(HERE, "results", "RESULTS.md")
# Raw estimates stashed by the analyses for figures.py (--figures).
ART: dict = {}

TAB3_REGS = ["Savings Percentage", "Human Capital Proxy", "Labor in research",
             "PIB towards research", "Patents per capita",
             "Labor not in research"]
THESIS_RD = ["PIB towards research", "Labor in research",
             "Patents per capita", "Human Capital Proxy"]
LEVEL_RD = ["RD_pct_GDP", "Researchers_per_1000_emp", "Patents_per_million",
            "Tertiary_share"]


# ── formatting ─────────────────────────────────────────────────────────────
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pstar(p: float) -> str:
    return f"{p:.3f}{stars(p)}" if np.isfinite(p) else "—"


def groups_apriori(df):
    return pd.Series({c: ("Innovative" if c.lower() in INNOVATIVE_CLUSTER
                          else "Emerging") for c in df["Country"].unique()})


# ── shared analyses ────────────────────────────────────────────────────────
def load_screened(path):
    raw = (pd.read_excel(path) if path.endswith(".xlsx")
           else pd.read_csv(path))
    raw = raw[raw["Year"] >= 2000] if "Year" in raw else raw
    res = outliers.screen(raw)                 # the Tab 1 default
    return raw, res


def tab3_fe(df, dep="Y by L", regs=TAB3_REGS):
    """Tab 3 defaults: log-transform, two-way FE, clustered by country."""
    regs = [r for r in regs if r in df.columns]
    d = df[["Country", "Year", dep] + regs].dropna().copy()
    for c in [dep] + regs:
        if (d[c] > 0).all():
            d[c] = np.log(d[c])
        elif (d[c] + 1 > 0).all():
            d[c] = np.log1p(d[c].clip(lower=0))
    d = d.set_index(["Country", "Year"])
    X = sm.add_constant(d[regs])
    fe = PanelOLS(d[dep], X, entity_effects=True, time_effects=True,
                  drop_absorbed=True).fit(cov_type="clustered",
                                          cluster_entity=True)
    return fe, d, regs


def hausman(d, dep, regs):
    X = sm.add_constant(d[regs])
    fe = PanelOLS(d[dep], X, entity_effects=True,
                  drop_absorbed=True).fit()
    re = RandomEffects(d[dep], X).fit()
    common = [r for r in regs if r in fe.params.index]
    b = fe.params[common] - re.params[common]
    V = fe.cov.loc[common, common] - re.cov.loc[common, common]
    H = float(b @ np.linalg.pinv(V.to_numpy()) @ b)
    return H, len(common), float(1 - stats.chi2.cdf(H, len(common)))


def fe_by_group(df, groups, focus, dep="Y by L", regs=TAB3_REGS):
    rows = []
    for name, sub in [("All", df)] + [
            (g, df[df["Country"].map(groups) == g])
            for g in ("Innovative", "Emerging")]:
        fe, _, _ = tab3_fe(sub, dep, regs)
        for v in focus:
            if v in fe.params:
                rows.append({"Sample": name, "Variable": v,
                             "Coef": fe.params[v], "SE": fe.std_errors[v],
                             "p": pstar(fe.pvalues[v]),
                             "N": int(fe.nobs)})
    return pd.DataFrame(rows)


def lag_profile_raw(df, max_lag=4):
    """Two-way FE coefficient (and SE, p) on R&D growth lagged L years, by
    group (Tab 3 spec with the R&D regressor replaced by its L-th lag)."""
    df = df.sort_values(["Country", "Year"])
    g = groups_apriori(df)
    rows = []
    for name, sub in [("All", df)] + [
            (k, df[df["Country"].map(g) == k]) for k in ("Innovative",
                                                         "Emerging")]:
        for L in range(max_lag + 1):
            d = sub.copy()
            d["RD_lag"] = d.groupby("Country")["PIB towards research"].shift(L)
            regs = ["RD_lag"] + [r for r in TAB3_REGS if r in d.columns
                                 and r != "PIB towards research"]
            fe, _, _ = tab3_fe(d, "Y by L", regs)
            rows.append({"Sample": name, "L": L,
                         "coef": fe.params["RD_lag"],
                         "se": fe.std_errors["RD_lag"],
                         "p": fe.pvalues["RD_lag"]})
    return pd.DataFrame(rows)


def _tab(name, df, caption, label):
    """Register *df* for LaTeX export and return it unchanged."""
    ART.setdefault("tables", {})[name] = (df, caption, label)
    return df


def lag_profile(df, max_lag=4, key=None):
    raw = lag_profile_raw(df, max_lag)
    if key:
        ART[key] = raw
    wide = raw.assign(v=[f"{c:+.3f}{stars(p)}" for c, p in
                         zip(raw.coef, raw.p)]).pivot(
        index="Sample", columns="L", values="v")
    wide.columns = [f"L{c}" for c in wide.columns]
    return wide.loc[["All", "Innovative", "Emerging"]].reset_index()


def kmeans_typology(df, cols):
    cm = df.groupby("Country")[cols].mean().dropna()
    X = StandardScaler().fit_transform(cm)
    lab = KMeans(2, random_state=42, n_init=10).fit_predict(X)
    ap = groups_apriori(df)
    # label the cluster holding more a-priori Innovative countries
    s = pd.Series(lab, index=cm.index)
    inno_id = s[ap.reindex(cm.index) == "Innovative"].mode().iloc[0]
    match = ((s == inno_id) == (ap.reindex(cm.index) == "Innovative")).mean()
    return {"silhouette": silhouette_score(X, lab),
            "agreement": match,
            "innovative": sorted(cm.index[s == inno_id]),
            "emerging": sorted(cm.index[s != inno_id])}


def club_lines(res):
    out = [f"Full-sample log-t: t = {res.full.t:.2f} "
           f"({'converges' if res.full.converges else 'rejects convergence'})"]
    for k, (m, t) in enumerate(zip(res.clubs, res.club_tests), 1):
        out.append(f"- Club {k} (t = {t.t:.2f}): {', '.join(m)}")
    if res.divergent:
        out.append(f"- Non-convergent: {', '.join(res.divergent)}")
    return "\n".join(out)


def ml_cv(df, target, feats):
    d = df[["Country", "Year", target] + feats].dropna().reset_index(drop=True)
    rows = []
    for name, m in (("Ridge", Ridge()),
                    ("Random Forest", RandomForestRegressor(
                        300, max_depth=8, random_state=42, n_jobs=-1))):
        for sch in ml_eval.SCHEMES[:2]:
            r = ml_eval.evaluate(m, d[feats], d[target], d["Country"],
                                 d["Year"], sch, 5)
            rows.append({"Model": name, "Scheme": sch,
                         "CV R²": r["cv_r2_mean"],
                         "Random-CV R² (leaky)": r["random_cv_r2"]})
    return pd.DataFrame(rows)


def gmm_section(df):
    """B13: reverse-causality-robust estimates (system GMM, R&D endogenous)."""
    d = df.copy()
    d["ly"] = 100 * d["log_Y_per_worker"]
    g = groups_apriori(d)
    em = (d["Country"].map(g) == "Emerging").astype(float)
    ctrls = ["Savings_rate", "Tertiary_share"]
    rows = []
    for x in ("log_RD_stock_per_worker", "RD_pct_GDP"):
        for yl in (1, 2):
            for lags in ((2, 3), (3, 4), (3, 5)):
                r = gmm.estimate(d, "ly", endog=[x], predet=ctrls,
                                 method="system", lags=lags, y_lags=yl)
                valid = (r.ar1_p < 0.10 and r.ar2_p > 0.05
                         and 0.05 < r.hansen_p < 0.99
                         and r.n_instruments <= r.n_groups)
                lr, lrse = r.long_run[x]
                ART.setdefault("gmm", []).append(
                    {"measure": x,
                     "label": f"{'stock' if x.startswith('log') else 'R&D % GDP'}"
                              f", {yl} y-lag{'s' if yl > 1 else ''}, "
                              f"Z lags {lags[0]}-{lags[1]}",
                     "beta": r.params[x], "se": r.se[x], "valid": valid})
                rows.append({"R&D measure": x, "y lags": yl,
                             "instr. lags": f"{lags[0]}-{lags[1]}",
                             "β R&D": r.params[x], "p": pstar(r.pvalues[x]),
                             "long-run": lr, "LR SE": lrse,
                             "AR(2) p": r.ar2_p, "Hansen p": r.hansen_p,
                             "#Z": r.n_instruments,
                             "valid": "✓" if valid else "✗",
                             "preferred": "★" if (yl == 2 and lags == (2, 3))
                             else ""})
    grid = pd.DataFrame(rows)
    het = []
    for x in ("log_RD_stock_per_worker", "RD_pct_GDP"):
        dd = d.assign(**{f"{x}×Emerging": d[x] * em})
        r = gmm.estimate(dd, "ly", endog=[x, f"{x}×Emerging"], predet=ctrls,
                         method="system", lags=(2, 3), y_lags=2)
        het.append({"R&D measure": x,
                    "β Innovative": r.params[x],
                    "p (Inn.)": pstar(r.pvalues[x]),
                    "Δ Emerging": r.params[f"{x}×Emerging"],
                    "p (Δ)": pstar(r.pvalues[f"{x}×Emerging"]),
                    "AR(2) p": r.ar2_p, "Hansen p": r.hansen_p})
    # Nickell-biased dynamic FE for comparison
    dd = d.sort_values(["Country", "Year"]).copy()
    gg = dd.groupby("Country")["ly"]
    dd["L1"], dd["L2"] = gg.shift(1), gg.shift(2)
    fes = []
    for x in ("log_RD_stock_per_worker", "RD_pct_GDP"):
        e = dd.dropna(subset=["ly", "L1", "L2", x] + ctrls).set_index(
            ["Country", "Year"])
        f = PanelOLS(e["ly"], e[["L1", "L2", x] + ctrls], entity_effects=True,
                     time_effects=True).fit(cov_type="clustered",
                                            cluster_entity=True)
        fes.append(f"{x}: β = {f.params[x]:+.3f} (p = {pstar(f.pvalues[x])})")
    T_ = ART.setdefault("tables", {})
    T_["tab_gmm_grid"] = (grid, "System GMM with R&D endogenous: "
                          "specification grid and diagnostics", "tab:gmm")
    T_["tab_gmm_heterogeneity"] = (
        pd.DataFrame(het), "System GMM, preferred specification: R&D "
        "effect by typology", "tab:gmmhet")
    return ["**B13 · Reverse causality: system GMM with R&D endogenous "
            "(Tab 14 → Dynamic GMM):**\n",
            "y = 100·log output per worker; R&D instrumented with its own "
            "lags (collapsed);\nsavings and tertiary share predetermined; "
            "two-step, Windmeijer SEs; year effects\ndemeaned. *valid* = "
            "AR(1) rejects, AR(2) does not, Hansen J in (0.05, 0.99), "
            "instruments ≤ groups.\n", md_table(grid), "",
            "Heterogeneity in the preferred specification (★):\n",
            md_table(pd.DataFrame(het)), "",
            "Dynamic two-way FE with the same regressors (Nickell-biased, R&D "
            "treated as exogenous): " + "; ".join(fes) + ".\n"]


# ── tracks ─────────────────────────────────────────────────────────────────
def track_a(quick):
    L = ["## Track A — thesis panel (`panel_data.xlsx`)\n",
         "Growth indices (previous year = 100) as submitted. Reproduces the "
         "thesis workflow;\nsee `docs/DATA_AUDIT.md` for the known issues "
         "(notably `Human Capital Proxy`).\n"]
    raw, scr = load_screened(THESIS)
    df = scr.df
    L.append(f"**A1 · Z-score screen (Tab 1 default, |z| > 3 on country "
             f"means):** excluded {', '.join(scr.flagged_countries) or 'none'}"
             f" → {df['Country'].nunique()} countries, {len(df)} obs.\n")
    fe, d, regs = tab3_fe(df)
    t = pd.DataFrame({"Variable": fe.params.index, "Coef": fe.params.values,
                      "SE": fe.std_errors.values,
                      "p": [pstar(p) for p in fe.pvalues.values]})
    H, k, pH = hausman(d, "Y by L", regs)
    L += ["**A2 · Panel FE (Tab 3 defaults: log, entity + time FE, clustered "
          f"SE), N = {int(fe.nobs)}:**\n", md_table(t),
          f"\nHausman FE vs RE: χ²({k}) = {H:.2f}, p = {pstar(pH)}\n"]
    gA = fe_by_group(df, groups_apriori(df),
                     ["PIB towards research", "Savings Percentage"])
    ART.setdefault("tables", {})["tab_fe_by_group_trackA"] = (
        gA, "Two-way FE by typology, thesis panel", "tab:feA")
    L += ["**A3 · GERD paradox — FE by a-priori group (Tab 3 on Tab 1 "
          "filters):**\n", md_table(gA), ""]
    L += ["**A3b · Timing — coefficient on R&D growth lagged L years "
          "(two-way FE):**\n", md_table(_tab("tab_lag_profile", lag_profile(df, key="lags_A"),
                                     "FE coefficient on R&D growth lagged L "
                                     "years (thesis panel)", "tab:lags")),
          "\nSame-year R&D growth is negative for the Innovative group; "
          "lags 1-3 turn positive\n(a J-curve), so the sign depends on "
          "timing — see *Methodology → What reproduces*.\n"]
    km = kmeans_typology(df, THESIS_RD)
    L.append(f"**A4 · K-Means, K = 2, thesis R&D variables (Tab 5):** "
             f"silhouette {km['silhouette']:.2f}, agreement with the thesis "
             f"typology {km['agreement']:.0%}.\n"
             f"- Cluster A: {', '.join(km['innovative'])}\n"
             f"- Cluster B: {', '.join(km['emerging'])}\n")
    lx = clubs.prepare_panel(df, "Y by L", chain=True)
    L += ["**A5 · Phillips-Sul clubs, chained `Y by L` (Tab 4):**\n",
          club_lines(clubs.club_clustering(lx)), ""]
    feats = [c for c in TAB3_REGS if c in df.columns]
    L += ["**A6 · ML, honest vs leaky CV (Tab 6), target `Y by L`, core "
          "features:**\n", md_table(ml_cv(df, "Y by L", feats)), ""]
    return "\n".join(L)


def beta_convergence(df):
    """C1: cross-country absolute β-convergence by typology.
    growth_i = a + β·log y_i,2000 + e_i, growth = average annual % growth
    of output per worker 2000-2023; HC1 SEs. Implied speed
    λ = −ln(1 + β·T/100)/T."""
    d = df[df["Year"].isin([2000, 2023])].pivot_table(
        index="Country", columns="Year", values="log_Y_per_worker").dropna()
    T = 23
    cs = pd.DataFrame({"Country": d.index, "initial": d[2000].values,
                       "growth": (d[2023] - d[2000]).values / T * 100})
    cs["Group"] = cs["Country"].map(groups_apriori(df))
    rows, fits = [], {}
    for name, sub in [("All", cs)] + [(k, cs[cs.Group == k])
                                      for k in ("Innovative", "Emerging")]:
        f = sm.OLS(sub["growth"], sm.add_constant(sub["initial"])).fit(
            cov_type="HC1")
        b = f.params["initial"]
        lam = (-np.log(1 + b * T / 100) / T * 100
               if b * T / 100 > -1 else np.nan)
        fits[name] = {"a": f.params["const"], "beta": b,
                      "p": f.pvalues["initial"]}
        rows.append({"Sample": name, "β": b, "SE": f.bse["initial"],
                     "p": pstar(f.pvalues["initial"]),
                     "speed λ (%/yr)": lam, "R²": f.rsquared,
                     "countries": len(sub)})
    ART["beta"] = (cs, fits)
    t = pd.DataFrame(rows)
    ART.setdefault("tables", {})["tab_beta_convergence"] = (
        t, "Absolute beta-convergence of output per worker by typology, "
           "2000-2023", "tab:beta")
    return ["**C1 · β-convergence by typology (Tab 4 → Absolute "
            "β-Convergence on Tab 1 group filters):**\n", md_table(t),
            "\nNegative β = initially poorer countries grew faster. With "
            "8-15 countries per group\nthe within-group slopes are "
            "imprecise; the frontier-gap coefficient (B7) is the\npanel "
            "counterpart.\n"]


def policy_section(df, nb):
    """D: R&D tax-incentive reforms and fiscal balance (policy_data.py)."""
    L = ["## Track B, part D — policy variables (`data/raw/policy/`)\n",
         "R&D tax generosity = OECD implied tax subsidy rate (1 − B-index), "
         "large profitable firm;\nfiscal stance = general government net "
         "lending, % of GDP (Eurostat).\n"]
    cols = [c for c in ("GBARD_mEUR", "Gov_balance_pct_GDP",
                        "RD_subsidy_large_profit") if c in df]
    cov = policy_data.coverage(df, sorted(df["Country"].unique()), cols,
                               window=(2000, 2023))
    summ = []
    for c in cols:
        full = cov[c].str.fullmatch(r"2000–2023")
        gaps = cov.loc[~full, ["Country", c]]
        summ.append({"Variable": c, "complete 2000-2023": int(full.sum()),
                     "countries": len(cov),
                     "incomplete": "; ".join(f"{a}: {b}" for a, b in
                                            gaps.itertuples(index=False))
                                   or "—"})
    L += ["**D1 · Coverage, 2000-2023** (full table: "
          "`data/policy_coverage.csv`):\n", md_table(pd.DataFrame(summ)),
          "\n`GBARD_mEUR` 2008-2016 is missing from the current NABS 2007 "
          "export (default view = last\n10 years); GBARD analyses wait for "
          "the full-period file.\n"]
    if "RD_tax_reform_year" not in df:
        return L
    ev = (df.dropna(subset=["RD_tax_reform_year"]).groupby("Country")
          ["RD_tax_reform_year"].first().astype(int).to_dict())
    L.append("**D2 · R&D tax-incentive reforms: staggered event study "
             "(Tab 14 → Events: R&D tax reforms):**\n")
    L.append("Reform = first year the large-firm subsidy rate rises ≥ 5 pp "
             "and stays up ≥ 2 more years.\nDetected: "
             + ", ".join(f"{c} {y}" for c, y in sorted(ev.items(),
                                                        key=lambda t: t[1]))
             + ".\n")
    rows = []
    for out in ("RD_pct_GDP", "Y_per_worker"):
        for ctrl in causal.CONTROL_GROUPS:
            for dt in (False, True):
                r = causal.event_study(df, out, ev, 3, 5, min(nb, 199),
                                       control=ctrl, detrend=dt)
                if out == "RD_pct_GDP" and ctrl == causal.CONTROL_GROUPS[1] \
                        and not dt:
                    ART["tax_es"] = {"R&D intensity": r}
                if out == "Y_per_worker" and ctrl == causal.CONTROL_GROUPS[1] \
                        and not dt:
                    ART.setdefault("tax_es", {})["Output per worker"] = r
                rows.append({"Outcome": out, "Controls": ctrl,
                             "Detrended": dt, "Post ATT %": r.overall_post,
                             "SE": r.overall_post_se,
                             "Pre-trend p": pstar(r.pretrend_p)})
    t = pd.DataFrame(rows)
    ART.setdefault("tables", {})["tab_tax_reform_event_study"] = (
        t, "R&D tax-incentive reforms: staggered event study", "tab:taxes")
    L += [md_table(t), "\nPre-trends are not significant for either "
          "outcome, so parallel trends are plausible\n(unlike EU accession, "
          "B9). R&D intensity rises after reforms but imprecisely;\noutput "
          "per worker does not move.\n"]
    d = df.copy()
    d["subsidy_pp"] = 100 * d["RD_subsidy_large_profit"]
    rows = []
    for out in ("RD_pct_GDP", "Y_per_worker"):
        lp = causal.local_projections(d, out, "subsidy_pp", 5, 1)
        for _, r in lp.iterrows():
            rows.append({"Outcome": out, "h": int(r.h), "β (% per pp)": r.beta,
                         "SE": r.se, "p": pstar(r.p)})
    L += ["**D3 · Local projections of a 1 pp change in the tax subsidy "
          "rate:**\n", md_table(pd.DataFrame(rows)),
          "\nContinuous subsidy changes are followed by *lower* R&D "
          "intensity after 3-5 years — consistent\nwith governments raising "
          "support when R&D is weak (policy endogeneity). The event study\n"
          "(D2), with testable pre-trends, is the preferred design.\n"]
    rows = []
    for out in ("Y_per_worker", "GDP_pc_2015usd", "Employment", "RD_pct_GDP"):
        lp = causal.local_projections(df, out, "Gov_balance_pct_GDP", 5, 1)
        rec = {"Outcome": out}
        for _, r in lp.iterrows():
            rec[f"h={int(r.h)}"] = f"{r.beta:+.2f}{stars(r.p)}"
        rows.append(rec)
    L += ["**D4 · Local projections of a 1 pp improvement in the "
          "government balance:**\n", md_table(pd.DataFrame(rows)),
          "\nOutput per worker and GDP per capita rise after the balance "
          "improves while employment does\nnot move, so this is not a "
          "labour-shedding artefact. But the *headline* balance improves\n"
          "automatically in booms, so these responses mix fiscal policy with "
          "the business cycle.\nA causal fiscal-consolidation test needs the "
          "cyclically adjusted balance (AMECO `UBLGAP`).\n"]
    return L


def make_figures(root):
    """Figures (PNG + PDF) and LaTeX tables from the stashed estimates."""
    import figures as F
    fdir, tdir = os.path.join(root, "figures"), os.path.join(root, "tables")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    F._style()
    made = []
    if "typology" in ART:
        df, _ = ART["typology"]
        made.append(F.fig_typology(df, LEVEL_RD, groups_apriori(df), fdir))
    if "clubs" in ART:
        made.append(F.fig_clubs(*ART["clubs"], fdir))
    if "lags_A" in ART:
        prof = {"Track A — thesis panel": ART["lags_A"]}
        if "lags_B" in ART:
            prof["Track B — rebuilt panel"] = ART["lags_B"]
        made.append(F.fig_jcurve(prof, fdir))
    if "events" in ART:
        made.append(F.fig_event_study(ART["events"], fdir))
    if "gmm" in ART:
        made.append(F.fig_gmm(pd.DataFrame(ART["gmm"]), fdir))
    if "dml" in ART:
        made.append(F.fig_dml(pd.DataFrame(ART["dml"]), fdir))
    if "beta" in ART:
        made.append(F.fig_beta(*ART["beta"], fdir))
    if "tax_es" in ART:
        made.append(F.fig_tax_reforms(ART["tax_es"], fdir))
    for name, (df_, cap, lab) in ART.get("tables", {}).items():
        with open(os.path.join(tdir, f"{name}.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write(F.to_latex(df_, cap, lab))
    return made


def track_b(quick):
    nb, ns = (29, 50) if quick else (299, 300)
    L = ["## Track B — rebuilt level panel (`data/panel_levels.csv`)\n",
         "Rebuilt from the raw workbook with the audit corrections; adds "
         "level variables.\n"]
    raw, scr = load_screened(LEVELS)
    df = scr.df
    hits = "; ".join(f"{c} ({', '.join(f'{v} z={z:+.1f}' for v, z in h)})"
                     for c, h in scr.flagged_countries.items())
    L.append(f"**B1 · Z-screen:** excluded {hits or 'none'} → "
             f"{df['Country'].nunique()} countries.\n")
    km = kmeans_typology(df, LEVEL_RD)
    ART["typology"] = (df, km)
    ART["lags_B"] = lag_profile_raw(df)
    L.append(f"**B2 · K-Means, K = 2, LEVEL R&D variables (Tab 5 default "
             f"on this panel):** silhouette {km['silhouette']:.2f}, agreement "
             f"with the thesis typology {km['agreement']:.0%}.\n"
             f"- Innovative: {', '.join(km['innovative'])}\n"
             f"- Emerging: {', '.join(km['emerging'])}\n")
    L += ["**B3 · GERD paradox on corrected inputs (same FE spec):**\n",
          md_table(fe_by_group(df, groups_apriori(df),
                               ["PIB towards research", "Human Capital Proxy",
                                "Savings Percentage"])), ""]
    lx = clubs.prepare_panel(df, "Y_per_worker", chain=False)
    ART["clubs"] = (lx, clubs.club_clustering(lx))
    L += ["**B4 · Phillips-Sul clubs on log output per worker levels "
          "(Tab 4, chain unticked):**\n", club_lines(clubs.club_clustering(lx)),
          ""]
    rows = []
    for c in ("log_Y_per_worker", "RD_pct_GDP", "Y by L"):
        r = panel_tests.cips(df, c, 1, False, n_sim=ns)
        rows.append({"Variable": c, "CIPS": r.stat, "5% cv": r.crit[0.05],
                     "p": pstar(r.p_value),
                     "Verdict": "stationary" if r.reject else "unit root"})
    L += ["**B5 · CIPS panel unit roots (Tab 10):**\n", md_table(pd.DataFrame(rows)),
          ""]
    rows = []
    for c, e in (("RD_pct_GDP", "log_Y_per_worker"),
                 ("log_Y_per_worker", "RD_pct_GDP")):
        for K in (1, 2):
            r = panel_tests.dumitrescu_hurlin(df, c, e, K, difference=True)
            rows.append({"X → Y (Δ)": f"{c} → {e}", "K": K, "W̄": r.W_bar,
                         "Z̃": r.Z_tilde, "p": pstar(r.p_Z_tilde)})
    L += ["**B6 · Dumitrescu-Hurlin panel Granger (Tab 11):**\n",
          md_table(pd.DataFrame(rows)), ""]
    fr = causal.frontier_regression(df, "Y_per_worker", "RD_pct_GDP",
                                    "Frontier_gap",
                                    ["Savings_rate", "Tertiary_share"])
    t = fr.table.assign(p=fr.table.p.map(pstar))
    L += [f"**B7 · Distance-to-frontier FE regression (Tab 14), N = "
          f"{fr.nobs}:**\n", md_table(t), ""]
    srt = df.sort_values(["Country", "Year"])
    lag = srt.groupby("Country")["Frontier_gap"].shift(1).reindex(df.index)
    state = (lag > df["Frontier_gap"].median()).astype(float).where(lag.notna())
    lp = causal.local_projections(df, "Y_per_worker", "RD_pct_GDP", 6, 2,
                                  state, ("Catch-up", "Near frontier"))
    lp = lp.assign(p=lp.p.map(pstar))[["h", "state", "beta", "se", "p"]]
    L += ["**B8 · Local projections, +1 pp R&D % GDP, state-dependent "
          "(Tab 14):**\n", md_table(lp), ""]
    rows = []
    for ctrl in causal.CONTROL_GROUPS:
        for dt in (False, True):
            r = causal.event_study(df, "Y_per_worker", causal.EU_ACCESSION,
                                   4, 8, nb, control=ctrl, detrend=dt)
            ART.setdefault("events", {})[(ctrl, dt)] = r
            rows.append({"Controls": ctrl, "Detrended": dt,
                         "Post ATT %": r.overall_post,
                         "SE": r.overall_post_se,
                         "Pre-trend": r.pretrend_mean,
                         "Pre-trend p": pstar(r.pretrend_p)})
    ART.setdefault("tables", {})["tab_event_study"] = (
        pd.DataFrame(rows), "EU accession event study (Callaway-Sant'Anna): "
        "control groups and pre-trend adjustment", "tab:event")
    L += ["**B9 · EU accession event study, Callaway-Sant'Anna (Tab 14):**\n",
          md_table(pd.DataFrame(rows)), ""]
    never = [c for c in df["Country"].unique()
             if c not in causal.EU_ACCESSION]
    sc = causal.synthetic_control(df, "Y_per_worker", "Poland", 2004, never)
    w = ", ".join(f"{c} {v:.2f}" for c, v in sc.weights[sc.weights > .01].items())
    L.append(f"**B10 · Synthetic control, Poland 2004 (Tab 14):** weights "
             f"{w}; average post gap {sc.avg_post_gap:+.1f} %; placebo "
             f"p = {sc.p_value:.3f}.\n")
    dm = causal.dml_plr(df, "Y_per_worker", "RD_pct_GDP",
                        ["Savings_rate", "Tertiary_share", "Frontier_gap"],
                        "Frontier_gap", groups_apriori(df))
    pz = 2 * (1 - stats.norm.cdf(abs(dm.cate_slope / dm.cate_se)))
    L += [f"**B11 · Double ML (Tab 14):** θ = {dm.theta:.3f} (SE "
          f"{dm.se:.3f}, p = {pstar(dm.p)}); ∂θ/∂gap = {dm.cate_slope:.3f} "
          f"(SE {dm.cate_se:.3f}, p = {pstar(pz)}).\n",
          md_table(dm.by_group.assign(p=dm.by_group.p.map(pstar))), ""]
    L += gmm_section(df)
    L += beta_convergence(df)
    if "RD_subsidy_large_profit" in df or "Gov_balance_pct_GDP" in df:
        L += policy_section(df, nb)
    rows = []
    d0 = pd.read_csv(LEVELS)
    grid = (("Luxembourg dropped, 1998+", ["Luxembourg"], 1998),
            ("Luxembourg dropped, 2000+", ["Luxembourg"], 2000),
            ("Lux. + Malta dropped, 2000+ (default)",
             ["Luxembourg", "Malta"], 2000),
            ("none dropped, 2000+", [], 2000))
    for name, excl, y0 in (grid[2:3] if quick else grid):
        d = d0[~d0["Country"].isin(excl) & (d0["Year"] >= y0)]
        for lr in ("Random Forest", "Lasso"):
            r = causal.dml_plr(d, "Y_per_worker", "RD_pct_GDP",
                               ["Savings_rate", "Tertiary_share",
                                "Frontier_gap"], "Frontier_gap",
                               groups_apriori(d), lr)
            pz = 2 * (1 - stats.norm.cdf(abs(r.cate_slope / r.cate_se)))
            ART.setdefault("dml", []).append(
                {"Sample": name, "Learner": lr, "slope": r.cate_slope,
                 "se": r.cate_se})
            gb = r.by_group.set_index("Group")["theta"]
            rows.append({"Sample": name, "Learner": lr, "θ": r.theta,
                         "θ p": pstar(r.p), "∂θ/∂gap": r.cate_slope,
                         "slope p": pstar(pz),
                         "θ Innovative": gb.get("Innovative", np.nan),
                         "θ Emerging": gb.get("Emerging", np.nan)})
    ART.setdefault("tables", {})["tab_dml_sensitivity"] = (
        pd.DataFrame(rows), "Double machine learning: sensitivity to sample "
        "and learner", "tab:dml")
    L += ["**B12 · DML sensitivity (sample × learner):**\n",
          md_table(pd.DataFrame(rows)),
          "\nThe gap slope is negative in most cells but its significance "
          "depends on the\nsample (Malta) and learner — report it as "
          "suggestive, not established.\n"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="also write results/figures (PNG, PDF) and "
                         "results/tables (LaTeX)")
    ap.add_argument("-o", "--out", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    parts = ["# Reproduced results\n",
             "Generated by `python reproduce.py"
             f"{' --quick' if a.quick else ''}`. Seeds are fixed; see the "
             "README section *Methodology* for how each result maps to the "
             "GUI.\n", track_a(a.quick)]
    if os.path.exists(LEVELS):
        parts.append(track_b(a.quick))
    else:
        parts.append("## Track B skipped\n\n`data/panel_levels.csv` not "
                     "found — run `python build_panel.py`.\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    if a.figures:
        made = make_figures(os.path.dirname(os.path.abspath(a.out)))
        print(f"figures: {', '.join(made)}")
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
