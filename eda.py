"""
eda.py — Exploratory statistical tests for country × year panels.

Pure functions returning DataFrames so the Statistics tab only formats and
plots. Every family of tests that is run across many variables at once
reports Benjamini-Hochberg FDR-adjusted p-values alongside raw ones.

Families
--------
  normality_tests        Shapiro-Wilk, Jarque-Bera, D'Agostino K², skew, kurt
  group_difference_tests Welch t, Mann-Whitney U, Kolmogorov-Smirnov, Levene,
                         Cohen's d — between two country groups
  country_effect_tests   One-way ANOVA and Kruskal-Wallis across countries
  trend_tests            Mann-Kendall on the cross-country yearly mean, OLS
                         slope, and share of countries with a significant
                         individual trend
  correlation_tests      Pearson & Spearman on pooled, between (country
                         means) and within (demeaned) variation
  cd_tests               Pesaran (2004) cross-sectional dependence per variable
  variance_decomposition Between vs. within share of total variance
  lead_lag_correlation   corr(x_{t-k}, y_t) on within-country demeaned data
  structural_break_tests Chow test on the yearly mean at candidate years
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Utilities ──────────────────────────────────────────────────────────────
def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (NaNs preserved)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def numeric_vars(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes("number").columns if c != "Year"]


def _demean(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[cols] - df.groupby("Country")[cols].transform("mean")


# ── Normality ──────────────────────────────────────────────────────────────
def normality_tests(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    rows = []
    for c in cols or numeric_vars(df):
        x = df[c].dropna().to_numpy()
        if len(x) < 8 or np.ptp(x) == 0:
            continue
        sw = stats.shapiro(x[:5000])
        jb = stats.jarque_bera(x)
        k2 = stats.normaltest(x)
        rows.append(dict(Variable=c, N=len(x), Skew=stats.skew(x),
                         Kurtosis=stats.kurtosis(x),
                         SW_W=sw.statistic, SW_p=sw.pvalue,
                         JB=jb.statistic, JB_p=jb.pvalue,
                         K2=k2.statistic, K2_p=k2.pvalue))
    out = pd.DataFrame(rows)
    for col in ("SW_p", "JB_p", "K2_p"):
        if len(out):
            out[col + "_fdr"] = bh_adjust(out[col])
    return out


# ── Group differences ──────────────────────────────────────────────────────
def group_difference_tests(df: pd.DataFrame, groups: pd.Series,
                           cols=None) -> pd.DataFrame:
    """Compare the two groups in *groups* (index: Country, values: label).

    Tests run on country-year observations; Cohen's d uses the pooled sd.
    """
    labels = [g for g in pd.unique(groups.dropna())][:2]
    if len(labels) < 2:
        return pd.DataFrame()
    gmap = df["Country"].map(groups)
    rows = []
    for c in cols or numeric_vars(df):
        a = df.loc[gmap == labels[0], c].dropna().to_numpy()
        b = df.loc[gmap == labels[1], c].dropna().to_numpy()
        if len(a) < 3 or len(b) < 3:
            continue
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        rows.append(dict(
            Variable=c, A=labels[0], B=labels[1],
            Mean_A=a.mean(), Mean_B=b.mean(),
            Cohen_d=(a.mean() - b.mean()) / sp if sp else np.nan,
            Welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue,
            MWU_p=stats.mannwhitneyu(a, b).pvalue,
            KS_p=stats.ks_2samp(a, b).pvalue,
            Levene_p=stats.levene(a, b).pvalue))
    out = pd.DataFrame(rows)
    for col in ("Welch_p", "MWU_p", "KS_p", "Levene_p"):
        if len(out):
            out[col + "_fdr"] = bh_adjust(out[col])
    return out


# ── Country heterogeneity ──────────────────────────────────────────────────
def country_effect_tests(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    rows = []
    for c in cols or numeric_vars(df):
        samples = [g[c].dropna().to_numpy() for _, g in df.groupby("Country")]
        samples = [s for s in samples if len(s) >= 2]
        if len(samples) < 2:
            continue
        f = stats.f_oneway(*samples)
        k = stats.kruskal(*samples)
        allx = np.concatenate(samples)
        ss_b = sum(len(s) * (s.mean() - allx.mean()) ** 2 for s in samples)
        ss_t = ((allx - allx.mean()) ** 2).sum()
        rows.append(dict(Variable=c, F=f.statistic, ANOVA_p=f.pvalue,
                         H=k.statistic, KW_p=k.pvalue,
                         Eta2=ss_b / ss_t if ss_t else np.nan))
    out = pd.DataFrame(rows)
    for col in ("ANOVA_p", "KW_p"):
        if len(out):
            out[col + "_fdr"] = bh_adjust(out[col])
    return out


# ── Trends ─────────────────────────────────────────────────────────────────
def trend_tests(df: pd.DataFrame, cols=None, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for c in cols or numeric_vars(df):
        ym = df.groupby("Year")[c].mean().dropna()
        if len(ym) < 5:
            continue
        tau = stats.kendalltau(ym.index, ym.values)
        lr = stats.linregress(ym.index, ym.values)
        sig_up = sig_dn = n = 0
        for _, g in df.groupby("Country"):
            s = g[["Year", c]].dropna()
            if len(s) < 5 or s[c].nunique() < 2:
                continue
            n += 1
            t = stats.kendalltau(s["Year"], s[c])
            if t.pvalue < alpha:
                sig_up += t.statistic > 0
                sig_dn += t.statistic < 0
        rows.append(dict(Variable=c, MK_tau=tau.statistic, MK_p=tau.pvalue,
                         Slope=lr.slope, Slope_p=lr.pvalue,
                         Countries_up=sig_up, Countries_down=sig_dn,
                         Countries_n=n))
    out = pd.DataFrame(rows)
    if len(out):
        out["MK_p_fdr"] = bh_adjust(out["MK_p"])
    return out


# ── Correlation (pooled / between / within) ────────────────────────────────
def correlation_tests(df: pd.DataFrame, y: str, cols=None) -> pd.DataFrame:
    cols = [c for c in (cols or numeric_vars(df)) if c != y]
    means = df.groupby("Country")[[y] + cols].mean()
    within = _demean(df, [y] + cols)
    rows = []
    for c in cols:
        rec = dict(Variable=c)
        for name, d in (("Pooled", df), ("Between", means), ("Within", within)):
            s = d[[y, c]].dropna()
            if len(s) < 4 or s[c].nunique() < 2 or s[y].nunique() < 2:
                continue
            pr = stats.pearsonr(s[c], s[y])
            sr = stats.spearmanr(s[c], s[y])
            rec.update({f"{name}_r": pr.statistic, f"{name}_p": pr.pvalue,
                        f"{name}_rho": sr.statistic})
        rows.append(rec)
    out = pd.DataFrame(rows)
    for name in ("Pooled", "Between", "Within"):
        if f"{name}_p" in out:
            out[f"{name}_p_fdr"] = bh_adjust(out[f"{name}_p"])
    return out


# ── Cross-sectional dependence ─────────────────────────────────────────────
def pesaran_cd(df: pd.DataFrame, col: str) -> tuple[float, float, float, int, int]:
    """Pesaran (2004) CD on within-country demeaned *col*.

    Returns (CD, p, mean |rho|, N, T).
    """
    wide = df.pivot_table(index="Year", columns="Country", values=col)
    wide = wide - wide.mean()
    wide = wide.loc[:, wide.std() > 0]
    N, T = wide.shape[1], wide.shape[0]
    if N < 3 or T < 3:
        return (np.nan,) * 3 + (N, T)
    rho = wide.corr(min_periods=3).to_numpy()
    iu = np.triu_indices(N, 1)
    r = rho[iu]
    r = r[np.isfinite(r)]
    cd = np.sqrt(2 * T / (N * (N - 1))) * r.sum()
    p = 2 * (1 - stats.norm.cdf(abs(cd)))
    return cd, p, np.abs(r).mean(), N, T


def cd_tests(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    rows = []
    for c in cols or numeric_vars(df):
        cd, p, ar, N, T = pesaran_cd(df, c)
        rows.append(dict(Variable=c, CD=cd, p=p, mean_abs_rho=ar, N=N, T=T))
    return pd.DataFrame(rows)


# ── Variance decomposition ─────────────────────────────────────────────────
def variance_decomposition(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    rows = []
    for c in cols or numeric_vars(df):
        s = df[["Country", c]].dropna()
        if s[c].var() == 0 or len(s) < 3:
            continue
        cm = s.groupby("Country")[c].transform("mean")
        between = ((cm - s[c].mean()) ** 2).sum()
        within = ((s[c] - cm) ** 2).sum()
        tot = between + within
        rows.append(dict(Variable=c, Between_share=between / tot,
                         Within_share=within / tot,
                         SD_between=s.groupby("Country")[c].mean().std(),
                         SD_within=(s[c] - cm).std()))
    return pd.DataFrame(rows)


# ── Lead-lag correlation ───────────────────────────────────────────────────
def lead_lag_correlation(df: pd.DataFrame, x: str, y: str,
                         max_lag: int = 5) -> pd.DataFrame:
    """corr(x_{t-k}, y_t) on within-country demeaned data, k = -L..L.

    Positive k: x leads y. Uses a 95 % band of ±1.96/√n.
    """
    d = df.sort_values(["Country", "Year"]).copy()
    d[[x, y]] = _demean(d, [x, y])
    rows = []
    for k in range(-max_lag, max_lag + 1):
        xs = d.groupby("Country")[x].shift(k)
        s = pd.concat([xs, d[y]], axis=1).dropna()
        if len(s) < 5:
            continue
        r, p = stats.pearsonr(s.iloc[:, 0], s.iloc[:, 1])
        rows.append(dict(Lag=k, r=r, p=p, n=len(s),
                         band=1.96 / np.sqrt(len(s))))
    return pd.DataFrame(rows)


# ── Structural breaks ──────────────────────────────────────────────────────
def chow_test(t: np.ndarray, y: np.ndarray, brk: float) -> tuple[float, float]:
    """Chow F-test for a break in a linear trend y = a + b·t at t = brk."""
    def ssr(tt, yy):
        X = np.column_stack([np.ones_like(tt), tt])
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        return ((yy - X @ beta) ** 2).sum()
    left, right = t < brk, t >= brk
    k = 2
    if left.sum() <= k or right.sum() <= k:
        return np.nan, np.nan
    s_p, s_1, s_2 = ssr(t, y), ssr(t[left], y[left]), ssr(t[right], y[right])
    dfd = len(t) - 2 * k
    F = ((s_p - s_1 - s_2) / k) / ((s_1 + s_2) / dfd)
    return F, 1 - stats.f.cdf(F, k, dfd)


def structural_break_tests(df: pd.DataFrame, cols=None,
                           years=(2004, 2008, 2009, 2013, 2020)) -> pd.DataFrame:
    """Chow tests on the cross-country yearly mean at candidate break years
    (EU enlargement 2004, GFC 2008-09, Croatia accession / post-crisis 2013,
    COVID 2020)."""
    rows = []
    for c in cols or numeric_vars(df):
        ym = df.groupby("Year")[c].mean().dropna()
        t, y = ym.index.to_numpy(float), ym.to_numpy(float)
        rec = dict(Variable=c)
        for yr in years:
            F, p = chow_test(t, y, yr)
            rec[f"F_{yr}"], rec[f"p_{yr}"] = F, p
        rows.append(rec)
    return pd.DataFrame(rows)
