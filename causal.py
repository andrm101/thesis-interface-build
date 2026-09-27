"""
causal.py — Causal and quasi-experimental designs for the country × year
panel (Tab 14). Pure functions: no Tk.

  frontier_regression   Two-way FE growth regression with R&D × distance-to-
                        frontier interaction (Griffith, Redding & Van Reenen
                        2004; Acemoglu, Aghion & Zilibotti 2006).
  local_projections     Jordà (2005) panel local projections of productivity
                        on an R&D change, linear or state-dependent.
  event_study           Callaway & Sant'Anna (2021) group-time ATTs with
                        never-treated controls, aggregated by event time,
                        country-cluster bootstrap.
  synthetic_control     Abadie, Diamond & Hainmueller (2010) with optional
                        pre-period demeaning (Ferman & Pinto 2021) and
                        in-space placebo inference.
  dml_plr               Double/debiased ML partially-linear model
                        (Chernozhukov et al. 2018), cross-fitted by country,
                        with a linear CATE in a moderator.

All designs take a long frame with `Country`, `Year` and the named columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import optimize, stats

# EU enlargement dates — the cleanest exogenous-timing events in the sample.
EU_ACCESSION = {
    "Cyprus": 2004, "Czechia": 2004, "Estonia": 2004, "Hungary": 2004,
    "Latvia": 2004, "Lithuania": 2004, "Malta": 2004, "Poland": 2004,
    "Slovakia": 2004, "Slovak Republic": 2004, "Slovenia": 2004,
    "Bulgaria": 2007, "Romania": 2007, "Croatia": 2013,
}


# ── helpers ────────────────────────────────────────────────────────────────
def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["Country", "Year"]).reset_index(drop=True)


def log_outcome(df: pd.DataFrame, col: str) -> pd.Series:
    """100 × log(col) — changes read as approximate percent."""
    return 100 * np.log(df[col].where(df[col] > 0))


def parse_events(text: str) -> dict[str, int]:
    """'Poland:2016, Czechia:2005' → {'Poland': 2016, 'Czechia': 2005}."""
    out = {}
    for part in text.replace(";", ",").split(","):
        if ":" in part:
            c, y = part.split(":", 1)
            if c.strip() and y.strip().isdigit():
                out[c.strip()] = int(y.strip())
    return out


def _panel_fit(d: pd.DataFrame, y: str, xs: list[str], time_fe=True):
    d = d.dropna(subset=[y] + xs).set_index(["Country", "Year"])
    mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=time_fe,
                   drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=True)


# ── 1. Distance to frontier ────────────────────────────────────────────────
@dataclass
class FrontierResult:
    table: pd.DataFrame
    marginal: pd.DataFrame           # dGrowth/dRD at gap quantiles
    nobs: int
    r2_within: float
    fit: object = field(repr=False, default=None)


def frontier_regression(df: pd.DataFrame, y_level: str, rd: str, gap: str,
                        controls: list[str] | None = None) -> FrontierResult:
    """growth_it = b1·RD_{t-1} + b2·Gap_{t-1} + b3·RD_{t-1}·Gap_{t-1}
                   + c'X_{t-1} + α_i + λ_t + e_it   (growth = Δ100·log y)

    b3 > 0: R&D pays off more far from the frontier (absorptive-capacity /
    catch-up channel, Griffith et al. 2004); b3 < 0: more near the frontier
    (innovation-based growth, Acemoglu et al. 2006).
    """
    d = _sorted(df)
    d["_ly"] = log_outcome(d, y_level)
    g = d.groupby("Country")
    d["growth"] = g["_ly"].diff()
    lag = lambda c: g[c].shift(1)  # noqa: E731
    d["RD_l1"], d["Gap_l1"] = lag(rd), lag(gap)
    d["RDxGap_l1"] = d["RD_l1"] * (d["Gap_l1"] - d["Gap_l1"].mean())
    xs = ["RD_l1", "Gap_l1", "RDxGap_l1"]
    for c in controls or []:
        d[f"{c}_l1"] = lag(c)
        xs.append(f"{c}_l1")
    fit = _panel_fit(d, "growth", xs)
    tab = pd.DataFrame({"Coef": fit.params, "SE": fit.std_errors,
                        "t": fit.tstats, "p": fit.pvalues}).reset_index(
        names="Variable")
    # Marginal effect of RD at gap quantiles (gap is centred in the product)
    gc = d["Gap_l1"].dropna()
    cov = fit.cov
    rows = []
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        z = gc.quantile(q) - gc.mean()
        me = fit.params["RD_l1"] + z * fit.params["RDxGap_l1"]
        var = (cov.loc["RD_l1", "RD_l1"] + z ** 2 * cov.loc["RDxGap_l1", "RDxGap_l1"]
               + 2 * z * cov.loc["RD_l1", "RDxGap_l1"])
        se = np.sqrt(max(var, 0))
        rows.append(dict(Gap_quantile=q, Gap=gc.quantile(q), dGrowth_dRD=me,
                         SE=se, p=2 * (1 - stats.norm.cdf(abs(me / se)))
                         if se else np.nan))
    return FrontierResult(tab, pd.DataFrame(rows), int(fit.nobs),
                          float(fit.rsquared_within), fit)


# ── 2. Local projections ───────────────────────────────────────────────────
def local_projections(df: pd.DataFrame, y_level: str, shock: str,
                      horizons: int = 6, n_lags: int = 2,
                      state: pd.Series | None = None,
                      state_names=("Catch-up", "Leader")) -> pd.DataFrame:
    """y_{t+h} − y_{t−1} = β_h·Δshock_t + Σ lags(Δshock, Δy) + α_i + λ_t.

    y = 100·log(y_level), so β_h is the % response of the outcome h years
    after a one-unit change in *shock*. If *state* (Country-Year aligned 0/1,
    1 = first state) is given, β_h is estimated separately per state.
    Returns rows (h, state, beta, se, lo, hi, n).
    """
    d = _sorted(df)
    d["_y"] = log_outcome(d, y_level)
    g = d.groupby("Country")
    d["_ds"] = g[shock].diff()
    d["_dy"] = g["_y"].diff()
    ctrl = []
    for l in range(1, n_lags + 1):
        d[f"_ds_l{l}"] = g["_ds"].shift(l)
        d[f"_dy_l{l}"] = g["_dy"].shift(l)
        ctrl += [f"_ds_l{l}", f"_dy_l{l}"]
    if state is not None:
        s = pd.Series(np.asarray(state, float), index=df.index)
        d["_S"] = s.reindex(_sorted(df.assign(_i=df.index))["_i"]).to_numpy()
        d["_shockA"] = d["_ds"] * d["_S"]
        d["_shockB"] = d["_ds"] * (1 - d["_S"])
        shocks = {"_shockA": state_names[0], "_shockB": state_names[1]}
    else:
        shocks = {"_ds": "All"}
    rows = []
    for h in range(horizons + 1):
        d["_lhs"] = g["_y"].shift(-h) - g["_y"].shift(1)
        fit = _panel_fit(d, "_lhs", list(shocks) + ctrl)
        for k, name in shocks.items():
            b, se = fit.params[k], fit.std_errors[k]
            rows.append(dict(h=h, state=name, beta=b, se=se,
                             lo=b - 1.96 * se, hi=b + 1.96 * se,
                             p=fit.pvalues[k], n=int(fit.nobs)))
    return pd.DataFrame(rows)


# ── 3. Staggered event study ───────────────────────────────────────────────
@dataclass
class EventStudyResult:
    by_event_time: pd.DataFrame       # e, att, se, lo, hi, n_cohorts
    group_time: pd.DataFrame          # g, t, e, att, n_treated
    overall_post: float
    overall_post_se: float
    pretrend_mean: float
    pretrend_p: float
    treated: list[str]
    controls: list[str]


CONTROL_GROUPS = ("Never treated", "Not yet treated")


def _att_gt(Y: pd.DataFrame, cohorts: dict[str, int], never: list[str],
            pre: int, post: int, not_yet: bool = False,
            detrend: bool = False) -> pd.DataFrame:
    rows = []
    years = Y.index
    for gyr in sorted(set(cohorts.values())):
        members = [c for c, v in cohorts.items() if v == gyr and c in Y]
        base = gyr - 1
        if not members or base not in years:
            continue
        for e in range(-pre, post + 1):
            t = gyr + e
            if t not in years or t == base:
                continue
            ctrl = list(never)
            if not_yet:     # units first treated after both t and the base
                ctrl += [c for c, v in cohorts.items()
                         if v > max(t, base) and c in Y]
            if not ctrl:
                continue
            dt = (Y.loc[t, members] - Y.loc[base, members]).mean()
            dc = (Y.loc[t, ctrl] - Y.loc[base, ctrl]).mean()
            if np.isfinite(dt) and np.isfinite(dc):
                rows.append(dict(g=gyr, t=t, e=e, att=dt - dc,
                                 n_treated=len(members)))
    out = pd.DataFrame(rows)
    if detrend and not out.empty:
        # Cohort-specific linear pre-trend through the base (e = −1, ATT 0),
        # extrapolated and removed from every event time.
        parts = []
        for gyr, s in out.groupby("g"):
            pre_s = s[s["e"] < -1]
            if len(pre_s) >= 2:
                e_all = np.r_[pre_s["e"], -1]
                a_all = np.r_[pre_s["att"], 0.0]
                slope, icpt = np.polyfit(e_all, a_all, 1)
                s = s.assign(att=s["att"] - (icpt + slope * s["e"]))
            parts.append(s)
        out = pd.concat(parts, ignore_index=True)
    return out


def _aggregate(gt: pd.DataFrame) -> pd.Series:
    if gt.empty:
        return pd.Series(dtype=float)
    return gt.groupby("e").apply(
        lambda s: np.average(s["att"], weights=s["n_treated"]),
        include_groups=False)


def event_study(df: pd.DataFrame, y_level: str, events: dict[str, int],
                pre: int = 4, post: int = 8, n_boot: int = 499,
                seed: int = 0, control: str = CONTROL_GROUPS[0],
                detrend: bool = False) -> EventStudyResult:
    """Callaway-Sant'Anna ATT(g, t) with a universal base period g−1 (so
    pre-period ATTs test parallel trends).

    control  "Never treated" or "Not yet treated" (adds later cohorts as
             controls until they are treated — e.g. 2007/2013 EU entrants
             for the 2004 cohort, which share its catch-up dynamics).
    detrend  remove a cohort-specific linear pre-trend (a sensitivity
             check when pre-trends are clearly non-zero; the adjusted
             pre-period ATTs are then zero-mean by construction, so the
             pre-trend test is reported on the unadjusted estimates).
    Inference: country-cluster bootstrap, resampling treated and control
    countries separately.
    """
    not_yet = control == CONTROL_GROUPS[1]
    d = _sorted(df)
    d["_y"] = log_outcome(d, y_level)
    Y = d.pivot_table(index="Year", columns="Country", values="_y")
    cohorts = {c: y for c, y in events.items() if c in Y.columns}
    never = [c for c in Y.columns if c not in cohorts]
    if not cohorts or (len(never) < 2 and not not_yet):
        raise ValueError("need ≥ 1 treated country in the data and ≥ 2 "
                         "never-treated controls")
    gt = _att_gt(Y, cohorts, never, pre, post, not_yet, detrend)
    point = _aggregate(gt)
    raw_pre = _aggregate(_att_gt(Y, cohorts, never, pre, post, not_yet))

    rng = np.random.default_rng(seed)
    treated = list(cohorts)
    boots, post_b, pre_b = [], [], []
    for _ in range(n_boot):
        tb = rng.choice(treated, len(treated), replace=True)
        cb = rng.choice(never, len(never), replace=True)
        # relabel draws so duplicates count as separate clusters
        cols, coh = {}, {}
        for k, c in enumerate(tb):
            cols[f"T{k}"] = Y[c]; coh[f"T{k}"] = cohorts[c]
        for k, c in enumerate(cb):
            cols[f"C{k}"] = Y[c]
        Yb = pd.DataFrame(cols)
        ctrl_b = [f"C{k}" for k in range(len(cb))]
        agg = _aggregate(_att_gt(Yb, coh, ctrl_b, pre, post, not_yet,
                                 detrend))
        raw = (agg if not detrend else
               _aggregate(_att_gt(Yb, coh, ctrl_b, pre, post, not_yet)))
        boots.append(agg.reindex(point.index))
        post_b.append(agg[agg.index >= 0].mean())
        pre_b.append(raw[raw.index < -1].mean() if (raw.index < -1).any()
                     else np.nan)
    B = pd.DataFrame(boots)
    se = B.std()
    tab = pd.DataFrame({"e": point.index, "att": point.values,
                        "se": se.values,
                        "lo": B.quantile(0.025).values,
                        "hi": B.quantile(0.975).values})
    tab["n_cohorts"] = gt.groupby("e")["g"].nunique().reindex(point.index).values
    post_pt = point[point.index >= 0].mean()
    pre_pts = raw_pre[raw_pre.index < -1]
    pre_mean = pre_pts.mean() if len(pre_pts) else np.nan
    pre_se = np.nanstd(pre_b)
    pre_p = (2 * (1 - stats.norm.cdf(abs(pre_mean / pre_se)))
             if pre_se and np.isfinite(pre_mean) else np.nan)
    return EventStudyResult(tab, gt, float(post_pt), float(np.nanstd(post_b)),
                            float(pre_mean), float(pre_p), treated, never)


# ── 4. Synthetic control ───────────────────────────────────────────────────
@dataclass
class SynthResult:
    treated: str
    year: int
    weights: pd.Series
    path: pd.DataFrame               # Year, actual, synthetic, gap
    pre_rmspe: float
    post_rmspe: float
    ratio: float
    placebo_ratios: pd.Series
    p_value: float
    avg_post_gap: float


def _sc_weights(y_pre: np.ndarray, X_pre: np.ndarray) -> np.ndarray:
    J = X_pre.shape[1]
    obj = lambda w: ((y_pre - X_pre @ w) ** 2).sum()  # noqa: E731
    res = optimize.minimize(obj, np.full(J, 1 / J), method="SLSQP",
                            bounds=[(0, 1)] * J,
                            constraints=({"type": "eq",
                                          "fun": lambda w: w.sum() - 1},),
                            options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def _sc_run(Y: pd.DataFrame, unit: str, donors: list[str], year: int,
            demean: bool):
    pre = Y.index < year
    y = Y[unit].to_numpy(float)
    X = Y[donors].to_numpy(float)
    if demean:
        y = y - y[pre].mean()
        X = X - X[pre].mean(axis=0)
    w = _sc_weights(y[pre], X[pre])
    synth = X @ w
    gap = y - synth
    pre_r = np.sqrt((gap[pre] ** 2).mean())
    post_r = np.sqrt((gap[~pre] ** 2).mean())
    return w, y, synth, gap, pre_r, post_r


def synthetic_control(df: pd.DataFrame, y_level: str, treated: str,
                      year: int, donors: list[str] | None = None,
                      demean: bool = True) -> SynthResult:
    d = _sorted(df)
    d["_y"] = log_outcome(d, y_level)
    Y = d.pivot_table(index="Year", columns="Country", values="_y").dropna(
        axis=1)
    if treated not in Y:
        raise ValueError(f"{treated} has missing outcome data")
    donors = [c for c in (donors or Y.columns) if c in Y and c != treated]
    if (Y.index < year).sum() < 3 or (Y.index >= year).sum() < 1:
        raise ValueError("need ≥ 3 pre-treatment years and ≥ 1 post year")
    w, y, s, gap, pre_r, post_r = _sc_run(Y, treated, donors, year, demean)
    ratio = post_r / pre_r if pre_r else np.inf
    plac = {}
    for p in donors:
        others = [c for c in donors if c != p]
        *_, pr, po = _sc_run(Y, p, others, year, demean)
        plac[p] = po / pr if pr else np.inf
    plac = pd.Series(plac)
    pval = (1 + (plac >= ratio).sum()) / (1 + len(plac))
    path = pd.DataFrame({"Year": Y.index, "actual": y, "synthetic": s,
                         "gap": gap})
    return SynthResult(treated, year,
                       pd.Series(w, index=donors).sort_values(ascending=False),
                       path, pre_r, post_r, ratio, plac, float(pval),
                       float(gap[Y.index >= year].mean()))


# ── 5. Double machine learning ─────────────────────────────────────────────
@dataclass
class DMLResult:
    theta: float
    se: float
    p: float
    n: int
    cate_slope: float | None
    cate_se: float | None
    moderator: str | None
    by_group: pd.DataFrame
    nuisance_r2: dict


def _cluster_ols(u: np.ndarray, V: np.ndarray, cl: np.ndarray):
    """OLS of u on V (no constant) with country-clustered sandwich SEs."""
    beta, *_ = np.linalg.lstsq(V, u, rcond=None)
    e = u - V @ beta
    bread = np.linalg.pinv(V.T @ V)
    meat = np.zeros((V.shape[1], V.shape[1]))
    for c in np.unique(cl):
        m = cl == c
        s = V[m].T @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(cl))
    adj = G / (G - 1) if G > 1 else 1.0
    cov = adj * bread @ meat @ bread
    return beta, np.sqrt(np.diag(cov))


def dml_plr(df: pd.DataFrame, y_level: str, treatment: str,
            controls: list[str], moderator: str | None = None,
            groups: pd.Series | None = None, learner: str = "Random Forest",
            n_folds: int = 5, seed: int = 0) -> DMLResult:
    """Partially linear model  growth = θ·D + g(X) + ε,  D = m(X) + v.

    growth = Δ100·log(y_level); D = treatment_{t−1}; X = lagged controls,
    lagged growth, country means of X (Mundlak) and year. Nuisances are
    cross-fitted with folds that hold out whole countries.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = _sorted(df)
    d["_ly"] = log_outcome(d, y_level)
    g = d.groupby("Country")
    d["_y"] = g["_ly"].diff()
    d["_D"] = g[treatment].shift(1)
    X_cols = ["Year"]
    d["_y_l1"] = g["_y"].shift(1)
    X_cols.append("_y_l1")
    for c in controls:
        d[f"{c}_l1"] = g[c].shift(1)
        d[f"{c}_cm"] = g[c].transform("mean")
        X_cols += [f"{c}_l1", f"{c}_cm"]
    if moderator:
        d["_M"] = g[moderator].shift(1)
    need = ["_y", "_D"] + X_cols + (["_M"] if moderator else [])
    d = d.dropna(subset=need).reset_index(drop=True)
    X = d[X_cols].to_numpy(float)
    yv, Dv = d["_y"].to_numpy(float), d["_D"].to_numpy(float)
    cl = d["Country"].to_numpy()

    def make():
        if learner.startswith("Lasso"):
            return make_pipeline(StandardScaler(), LassoCV(cv=3, random_state=seed))
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1)

    ry, rd = np.zeros_like(yv), np.zeros_like(Dv)
    k = max(2, min(n_folds, len(np.unique(cl))))
    for tr, te in GroupKFold(n_splits=k).split(X, groups=cl):
        ry[te] = yv[te] - make().fit(X[tr], yv[tr]).predict(X[te])
        rd[te] = Dv[te] - make().fit(X[tr], Dv[tr]).predict(X[te])
    r2 = {"outcome": 1 - (ry ** 2).sum() / ((yv - yv.mean()) ** 2).sum(),
          "treatment": 1 - (rd ** 2).sum() / ((Dv - Dv.mean()) ** 2).sum()}

    (theta,), (se,) = _cluster_ols(ry, rd[:, None], cl)
    cate_b = cate_se = None
    if moderator:
        z = d["_M"].to_numpy(float)
        z = z - z.mean()
        b, s = _cluster_ols(ry, np.column_stack([rd, rd * z]), cl)
        cate_b, cate_se = float(b[1]), float(s[1])

    rows = []
    if groups is not None:
        lab = d["Country"].map(groups)
        for gname in pd.unique(lab.dropna()):
            m = (lab == gname).to_numpy()
            if m.sum() > 10 and len(np.unique(cl[m])) > 1:
                (bt,), (st,) = _cluster_ols(ry[m], rd[m][:, None], cl[m])
                rows.append(dict(Group=gname, theta=bt, se=st,
                                 p=2 * (1 - stats.norm.cdf(abs(bt / st))),
                                 n=int(m.sum()),
                                 countries=len(np.unique(cl[m]))))
    return DMLResult(float(theta), float(se),
                     float(2 * (1 - stats.norm.cdf(abs(theta / se)))),
                     len(yv), cate_b, cate_se, moderator, pd.DataFrame(rows),
                     r2)
