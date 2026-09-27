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
from constants import INNOVATIVE_CLUSTER, stars

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.join(HERE, "panel_data.xlsx")
LEVELS = os.path.join(HERE, "data", "panel_levels.csv")
OUT = os.path.join(HERE, "results", "RESULTS.md")

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


def lag_profile(df, max_lag=4):
    """Two-way FE coefficient on R&D growth lagged L years, by group
    (Tab 3 spec with the R&D regressor replaced by its L-th lag)."""
    df = df.sort_values(["Country", "Year"])
    g = groups_apriori(df)
    rows = []
    for name, sub in [("All", df)] + [
            (k, df[df["Country"].map(g) == k]) for k in ("Innovative",
                                                         "Emerging")]:
        rec = {"Sample": name}
        for L in range(max_lag + 1):
            d = sub.copy()
            d["RD_lag"] = d.groupby("Country")["PIB towards research"].shift(L)
            regs = ["RD_lag"] + [r for r in TAB3_REGS if r in d.columns
                                 and r != "PIB towards research"]
            fe, _, _ = tab3_fe(d, "Y by L", regs)
            rec[f"L{L}"] = (f"{fe.params['RD_lag']:+.3f}"
                            f"{stars(fe.pvalues['RD_lag'])}")
        rows.append(rec)
    return pd.DataFrame(rows)


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
    L += ["**A3 · GERD paradox — FE by a-priori group (Tab 3 on Tab 1 "
          "filters):**\n",
          md_table(fe_by_group(df, groups_apriori(df),
                               ["PIB towards research", "Savings Percentage"])),
          ""]
    L += ["**A3b · Timing — coefficient on R&D growth lagged L years "
          "(two-way FE):**\n", md_table(lag_profile(df)),
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
            rows.append({"Controls": ctrl, "Detrended": dt,
                         "Post ATT %": r.overall_post,
                         "SE": r.overall_post_se,
                         "Pre-trend": r.pretrend_mean,
                         "Pre-trend p": pstar(r.pretrend_p)})
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
            gb = r.by_group.set_index("Group")["theta"]
            rows.append({"Sample": name, "Learner": lr, "θ": r.theta,
                         "θ p": pstar(r.p), "∂θ/∂gap": r.cate_slope,
                         "slope p": pstar(pz),
                         "θ Innovative": gb.get("Innovative", np.nan),
                         "θ Emerging": gb.get("Emerging", np.nan)})
    L += ["**B12 · DML sensitivity (sample × learner):**\n",
          md_table(pd.DataFrame(rows)),
          "\nThe gap slope is negative in most cells but its significance "
          "depends on the\nsample (Malta) and learner — report it as "
          "suggestive, not established.\n"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
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
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
