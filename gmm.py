"""
gmm.py — Dynamic panel GMM: Arellano-Bond (1991) difference GMM and
Blundell-Bond (1998) system GMM, following Roodman's (2009) xtabond2
conventions.

    y_it = ρ·y_i,t−1 + β'x_it + η_i + λ_t + ε_it

Why: fixed effects with a lagged dependent variable are biased (Nickell
1981), and R&D responds to productivity (Dumitrescu-Hurlin, Tab 11), so
R&D must be treated as endogenous. GMM instruments the first-differenced
equation with deeper lags of the levels (and, for system GMM, the level
equation with lagged differences).

Choices, all standard and reported with the results:
  • year effects removed by cross-sectional demeaning of every variable;
  • instruments COLLAPSED and lag-limited (Roodman 2009) — with N ≈ 24
    countries, uncollapsed instrument sets would exceed N and overfit;
  • two-step estimates with Windmeijer (2005) finite-sample corrected SEs
    (one-step cluster-robust SEs also available);
  • diagnostics: Hansen J over-identification test, Arellano-Bond AR(1)
    and AR(2) tests on differenced residuals (full variance including
    estimation error), instrument count vs. N.

Validated against pydynpd (the Python port of xtabond2): difference GMM
matches coefficients, Windmeijer SEs, Hansen J and AR tests; system GMM
matches ρ, β and their SEs to three decimals.

Variable roles (xtabond2 terminology):
  endogenous     x_t correlated with ε_t      → GMM lags 2..L (diff eq.),
                                                Δx_t−1 (level eq.)
  predetermined  x_t correlated with ε_t−1…   → lags 1..L, Δx_t
  exogenous      uncorrelated with ε at all t → IV-style, the variable itself
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GMMResult:
    method: str
    names: list[str]
    params: pd.Series
    se: pd.Series
    n_groups: int
    n_obs: int
    n_instruments: int
    hansen: float
    hansen_df: int
    hansen_p: float
    ar1: float
    ar1_p: float
    ar2: float
    ar2_p: float
    long_run: dict = field(default_factory=dict)   # var → (effect, se)
    notes: list[str] = field(default_factory=list)

    @property
    def pvalues(self) -> pd.Series:
        z = (self.params / self.se).abs().to_numpy()
        return pd.Series(2 * (1 - stats.norm.cdf(z)), index=self.params.index)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"Variable": self.params.index,
                             "Coef": self.params.values,
                             "SE": self.se.values,
                             "z": (self.params / self.se).values,
                             "p": self.pvalues.values})


# ── data preparation ───────────────────────────────────────────────────────
def _prepare(df, y, endog, predet, exog, time_demean):
    cols = [y] + endog + predet + exog
    d = df[["Country", "Year"] + cols].copy()
    if time_demean:
        d[cols] = d[cols] - d.groupby("Year")[cols].transform("mean")
    years = np.sort(d["Year"].unique())
    panels = {}
    for c, g in d.groupby("Country"):
        g = g.set_index("Year").reindex(years)
        panels[c] = {v: g[v].to_numpy(float) for v in cols}
    return panels, years


def _lag(a, l):
    """a[t − l] aligned to t (NaN where t − l < 0)."""
    out = np.full_like(a, np.nan)
    if l < len(a):
        out[l:] = a[:len(a) - l] if l > 0 else a
    return out


def _build_unit(p, y, endog, predet, exog, lags, system, y_lags=1):
    """Stacked (y, X, Z, eq-type, t) for one country.

    Regressors: [y_{t−1} … y_{t−y_lags}] + endog + predet + exog
    (+ const in system).
    """
    T = len(p[y])
    lo, hi = lags
    yv = p[y]
    xs = endog + predet + exog
    X_lvl = np.column_stack([_lag(yv, j) for j in range(1, y_lags + 1)]
                            + [p[v] for v in xs])
    rows_d, rows_l = [], []
    for t in range(y_lags + 1, T):
        # differenced equation at t
        dy = yv[t] - yv[t - 1]
        dx = X_lvl[t] - X_lvl[t - 1]
        z = []
        for l in range(lo, hi + 1):                    # y lags ≥ 2
            z.append(yv[t - l] if t - l >= 0 else np.nan)
        for v in endog:
            for l in range(lo, hi + 1):
                z.append(p[v][t - l] if t - l >= 0 else np.nan)
        for v in predet:
            for l in range(max(1, lo - 1), hi):
                z.append(p[v][t - l] if t - l >= 0 else np.nan)
        for v in exog:
            z.append(p[v][t] - p[v][t - 1])
        rows_d.append((t, dy, dx, np.array(z, float)))
        if system:
            zl = [yv[t - 1] - yv[t - 2]]                   # Δy_{t−1}
            for v in endog:
                zl.append(p[v][t - 1] - p[v][t - 2])       # Δx_{t−1}
            for v in predet:
                zl.append(p[v][t] - p[v][t - 1])           # Δx_t
            for v in exog:
                zl.append(p[v][t])
            zl.append(1.0)
            rows_l.append((t, yv[t], np.r_[X_lvl[t], 1.0], np.array(zl)))
    return rows_d, rows_l


def _assemble(panels, y, endog, predet, exog, lags, system, y_lags=1):
    units = []
    k = y_lags + len(endog) + len(predet) + len(exog) + (1 if system else 0)
    for c, p in panels.items():
        rd, rl = _build_unit(p, y, endog, predet, exog, lags, system, y_lags)
        ys, Xs, Zd, Zl, kind, ts = [], [], [], [], [], []
        for t, dy, dx, z in rd:
            if not (np.isfinite(dy) and np.isfinite(dx).all()):
                continue
            ys.append(dy)
            Xs.append(np.r_[dx, 0.0] if system else dx)
            Zd.append(np.nan_to_num(z)); kind.append(0); ts.append(t)
        for t, yl, xl, z in rl:
            if not (np.isfinite(yl) and np.isfinite(xl).all()):
                continue
            ys.append(yl); Xs.append(xl)
            Zl.append(np.nan_to_num(z)); kind.append(1); ts.append(t)
        if not ys:
            continue
        nd, nl = len(Zd), len(Zl)
        Ld = len(Zd[0]) if Zd else len(rd[0][3])
        Ll = len(Zl[0]) if Zl else (len(rl[0][3]) if rl else 0)
        Z = np.zeros((nd + nl, Ld + Ll))
        if nd:
            Z[:nd, :Ld] = np.vstack(Zd)
        if nl:
            Z[nd:, Ld:] = np.vstack(Zl)
        units.append(dict(country=c, y=np.array(ys), X=np.vstack(Xs).reshape(-1, k),
                          Z=Z, kind=np.array(kind), t=np.array(ts)))
    return units, k


def _H(u):
    """One-step weighting matrix (xtabond2 default h(3)) = covariance of
    the transformed errors under iid ε: Δε_t has variance 2 and covariance
    −1 with Δε_t±1; level ε_t has variance 1; Δε_t covaries +1 with ε_t
    and −1 with ε_t−1 (system GMM)."""
    n = len(u["y"])
    H = np.zeros((n, n))
    kind, t = u["kind"], u["t"]
    for a in range(n):
        for b in range(n):
            if kind[a] == 0 and kind[b] == 0:
                gap = abs(t[a] - t[b])
                H[a, b] = 2.0 if gap == 0 else (-1.0 if gap == 1 else 0.0)
            elif kind[a] == 1 and kind[b] == 1:
                H[a, b] = 1.0 if t[a] == t[b] else 0.0
            else:
                td, tl = (t[a], t[b]) if kind[a] == 0 else (t[b], t[a])
                H[a, b] = 1.0 if tl == td else (-1.0 if tl == td - 1 else 0.0)
    return H


# ── estimator ──────────────────────────────────────────────────────────────
def estimate(df: pd.DataFrame, y: str, endog: list[str] | None = None,
             predet: list[str] | None = None, exog: list[str] | None = None,
             method: str = "system", lags: tuple[int, int] = (2, 3),
             two_step: bool = True, time_demean: bool = True,
             y_lags: int = 1) -> GMMResult:
    """Difference (method="difference") or system (method="system") GMM.

    lags    GMM instrument lag range for y and endogenous x (predetermined
            x use one lag less). Raise the lower bound to 3 when AR(2)
            rejects: lag-2 levels are then correlated with Δε_t.
    y_lags  number of lags of y as regressors (2 when the errors are
            serially correlated because dynamics are too short).
    """
    endog, predet, exog = list(endog or []), list(predet or []), list(exog or [])
    system = method == "system"
    panels, _ = _prepare(df, y, endog, predet, exog, time_demean)
    units, k = _assemble(panels, y, endog, predet, exog, lags, system, y_lags)
    names = ([f"L{j}.{y}" for j in range(1, y_lags + 1)] + endog + predet
             + exog + (["const"] if system else []))
    # drop instrument columns that are identically zero across all units
    Zall = np.vstack([u["Z"] for u in units])
    keep = np.abs(Zall).sum(axis=0) > 0
    for u in units:
        u["Z"] = u["Z"][:, keep]
    L = int(keep.sum())
    N = len(units)

    ZX = sum(u["Z"].T @ u["X"] for u in units)
    Zy = sum(u["Z"].T @ u["y"] for u in units)
    W1 = np.linalg.pinv(sum(u["Z"].T @ _H(u) @ u["Z"] for u in units))
    A1 = ZX.T @ W1 @ ZX
    b1 = np.linalg.solve(A1, ZX.T @ W1 @ Zy)
    for u in units:
        u["e1"] = u["y"] - u["X"] @ b1
    S1 = sum(u["Z"].T @ np.outer(u["e1"], u["e1"]) @ u["Z"] for u in units)
    A1i = np.linalg.inv(A1)
    V1 = A1i @ ZX.T @ W1 @ S1 @ W1 @ ZX @ A1i          # robust one-step

    if two_step:
        W2 = np.linalg.pinv(S1)
        A2 = ZX.T @ W2 @ ZX
        A2i = np.linalg.inv(A2)
        b = A2i @ ZX.T @ W2 @ Zy
        for u in units:
            u["e"] = u["y"] - u["X"] @ b
        g2 = sum(u["Z"].T @ u["e"] for u in units)
        # Windmeijer (2005) correction
        D = np.zeros((k, k))
        for j in range(k):
            dS = -sum(u["Z"].T @ (np.outer(u["X"][:, j], u["e1"])
                                   + np.outer(u["e1"], u["X"][:, j])) @ u["Z"]
                      for u in units)
            # β2(W) with W = S⁻¹ ⇒ dβ2 = −A2⁻¹ X'Z W (dS) W Z'û2
            D[:, j] = -A2i @ ZX.T @ W2 @ dS @ W2 @ g2
        V = A2i + D @ A2i + A2i @ D.T + D @ V1 @ D.T
        Wj = W2
    else:
        b, V = b1, V1
        for u in units:
            u["e"] = u["e1"]
        Wj = np.linalg.pinv(S1)
    # Hansen J with the two-step (efficient) weight
    g = sum(u["Z"].T @ u["e"] for u in units)
    J = float(g @ Wj @ g)
    Jdf = L - k
    Jp = float(1 - stats.chi2.cdf(J, Jdf)) if Jdf > 0 else np.nan

    def ar(order):
        """Arellano-Bond m_j on differenced residuals with the full
        variance (Roodman 2009, eq. for m_j), including the terms for
        estimation error in β."""
        num = v1 = 0.0
        wX = np.zeros(k)
        wZ = np.zeros(L)
        for u in units:
            m = u["kind"] == 0
            e = u["e"][m]
            s_ = pd.Series(e, index=u["t"][m])
            w = np.nan_to_num(s_.reindex(u["t"][m] - order).to_numpy())
            wf = np.zeros(len(u["y"]))
            wf[m] = w
            num += w @ e
            v1 += (w @ e) ** 2
            wX += wf @ u["X"]
            wZ += u["Z"].T @ u["e"] * (u["e"] @ wf)
        Ab = np.linalg.inv(ZX.T @ Wj @ ZX)
        v2 = -2 * wX @ Ab @ ZX.T @ Wj @ wZ
        v3 = wX @ V @ wX
        var = v1 + v2 + v3
        z = num / np.sqrt(var) if var > 0 else np.nan
        return z, float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) \
            else np.nan

    a1, a1p = ar(1)
    a2, a2p = ar(2)
    params = pd.Series(b, index=names)
    se = pd.Series(np.sqrt(np.clip(np.diag(V), 0, None)), index=names)
    # long-run effects β / (1 − ρ), delta method
    lr = {}
    rho = b[:y_lags].sum()
    for j, nm in enumerate(names[y_lags:], y_lags):
        if nm == "const":
            continue
        eff = b[j] / (1 - rho)
        grad = np.zeros(k)
        grad[:y_lags] = b[j] / (1 - rho) ** 2
        grad[j] = 1 / (1 - rho)
        lr[nm] = (float(eff), float(np.sqrt(max(grad @ V @ grad, 0))))
    notes = [f"{'Two-step, Windmeijer SEs' if two_step else 'One-step, robust SEs'}"
             f"; instruments collapsed, lags {lags[0]}-{lags[1]}"
             f"{'; year effects demeaned' if time_demean else ''}"]
    if L > N:
        notes.append(f"⚠ {L} instruments > {N} groups: Hansen J is weakened "
                     "and estimates drift toward OLS/FE — reduce lags.")
    return GMMResult(
        "System GMM (Blundell-Bond)" if system else
        "Difference GMM (Arellano-Bond)", names, params, se, N,
        int(sum(len(u["y"]) for u in units)), L, J, Jdf, Jp, a1, a1p, a2, a2p,
        lr, notes)
