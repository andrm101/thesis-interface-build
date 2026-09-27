"""
panel_tests.py — Second-generation panel unit-root and causality tests.

  cips               Pesaran (2007) CIPS unit-root test: cross-sectionally
                     augmented Dickey-Fuller (CADF) regressions, robust to
                     a common factor (the cross-sectional dependence the
                     Pesaran CD test in Tab 2 detects). Critical values are
                     simulated for the panel's own N and T instead of read
                     from asymptotic tables.
  dumitrescu_hurlin  Dumitrescu & Hurlin (2012) panel Granger non-causality
                     test with heterogeneous coefficients: H0 x does not
                     Granger-cause y in ANY country; H1 it does in some.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    w = df.pivot_table(index="Year", columns="Country", values=col)
    return w.dropna(axis=1)            # balanced panel required


def _ols_t(y: np.ndarray, X: np.ndarray, k: int) -> float:
    """t-statistic of coefficient k in OLS of y on X."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return np.nan
    s2 = e @ e / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta[k] / np.sqrt(cov[k, k])


# ── CIPS ───────────────────────────────────────────────────────────────────
def _cadf_stats(Y: np.ndarray, p: int, trend: bool) -> np.ndarray:
    """Individual CADF t-statistics for a T × N array of levels."""
    T, N = Y.shape
    ybar = Y.mean(axis=1)
    dY, dbar = np.diff(Y, axis=0), np.diff(ybar)
    ts = np.empty(N)
    for i in range(N):
        rows_y, rows_X = [], []
        for t in range(p + 1, T):          # t indexes levels
            x = [1.0, Y[t - 1, i], ybar[t - 1], dbar[t - 1]]
            for j in range(1, p + 1):
                x += [dY[t - 1 - j, i], dbar[t - 1 - j]]
            if trend:
                x.append(float(t))
            rows_X.append(x)
            rows_y.append(dY[t - 1, i])
        ts[i] = _ols_t(np.asarray(rows_y), np.asarray(rows_X), 1)
    return ts


def _truncate(t: np.ndarray, trend: bool) -> np.ndarray:
    # Pesaran (2007) truncation bounds for the CIPS* statistic.
    k1, k2 = (6.42, 1.70) if not trend else (6.12, 4.16)
    return np.clip(t, -k1, k2)


@dataclass
class CIPSResult:
    variable: str
    stat: float
    crit: dict            # {0.01, 0.05, 0.10: value}
    p_value: float
    N: int
    T: int
    lags: int
    trend: bool
    individual: pd.Series

    @property
    def reject(self) -> bool:
        return self.stat < self.crit[0.05]


def cips(df: pd.DataFrame, col: str, lags: int = 1, trend: bool = False,
         n_sim: int = 500, seed: int = 0, log: bool = False) -> CIPSResult:
    """H0: every country's series has a unit root. Reject for very negative
    CIPS. Critical values and p-value come from *n_sim* simulated panels of
    independent random walks with a common factor, at this N and T."""
    W = _wide(df, col)
    Y = W.to_numpy(float)
    if log:
        Y = np.log(Y)
    T, N = Y.shape
    if N < 3 or T < lags + 8:
        raise ValueError(f"need a balanced panel with N ≥ 3 and T ≥ "
                         f"{lags + 8}; got N={N}, T={T}")
    ind = _cadf_stats(Y, lags, trend)
    stat = float(np.nanmean(_truncate(ind, trend)))
    rng = np.random.default_rng(seed)
    sims = np.empty(n_sim)
    for s in range(n_sim):
        f = np.cumsum(rng.normal(size=T))
        Ys = np.cumsum(rng.normal(size=(T, N)), axis=0) + \
            np.outer(f, rng.normal(size=N))
        sims[s] = np.nanmean(_truncate(_cadf_stats(Ys, lags, trend), trend))
    crit = {a: float(np.quantile(sims, a)) for a in (0.01, 0.05, 0.10)}
    return CIPSResult(col, stat, crit, float((sims <= stat).mean()), N, T,
                      lags, trend, pd.Series(ind, index=W.columns))


# ── Dumitrescu-Hurlin ──────────────────────────────────────────────────────
@dataclass
class DHResult:
    cause: str
    effect: str
    lags: int
    W_bar: float
    Z_bar: float
    p_Z_bar: float
    Z_tilde: float
    p_Z_tilde: float
    N: int
    T: int
    individual: pd.DataFrame      # Country, W, p


def dumitrescu_hurlin(df: pd.DataFrame, cause: str, effect: str,
                      lags: int = 1, difference: bool = False) -> DHResult:
    """Unit-by-unit Wald tests of lagged *cause* in an AR(lags) model of
    *effect*, averaged (W̄) and standardised: Z̄ (T → ∞ then N → ∞) and the
    small-T Z̃ (valid for T > 5 + 3·lags). Both series should be stationary
    — set difference=True for I(1) levels."""
    d = df.sort_values(["Country", "Year"])
    Wy, Wx = _wide(d, effect), _wide(d, cause)
    common = Wy.columns.intersection(Wx.columns)
    Wy, Wx = Wy[common], Wx[common]
    Wy, Wx = Wy.loc[Wy.index.intersection(Wx.index)], \
        Wx.loc[Wy.index.intersection(Wx.index)]
    if difference:
        Wy, Wx = Wy.diff().iloc[1:], Wx.diff().iloc[1:]
    K = lags
    Yv, Xv = Wy.to_numpy(float), Wx.to_numpy(float)
    T_all, N = Yv.shape
    T = T_all - K
    if N < 2 or T <= 2 * K + 1:
        raise ValueError(f"not enough data (N={N}, T={T_all})")
    rows = []
    for i, c in enumerate(common):
        y = Yv[K:, i]
        Xr = [np.ones(T)]
        Xr += [Yv[K - j:T_all - j, i] for j in range(1, K + 1)]
        Xu = Xr + [Xv[K - j:T_all - j, i] for j in range(1, K + 1)]
        Xr, Xu = np.column_stack(Xr), np.column_stack(Xu)
        er = y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        eu = y - Xu @ np.linalg.lstsq(Xu, y, rcond=None)[0]
        ssr_r, ssr_u = er @ er, eu @ eu
        dof = T - 2 * K - 1
        W = K * ((ssr_r - ssr_u) / K) / (ssr_u / dof)   # K·F = Wald
        rows.append(dict(Country=c, W=W, p=1 - stats.chi2.cdf(W, K)))
    ind = pd.DataFrame(rows)
    Wb = float(ind.W.mean())
    Zb = np.sqrt(N / (2 * K)) * (Wb - K)
    Zt = np.nan
    if T > 5 + 3 * K:
        Zt = (np.sqrt(N / (2 * K) * (T - 2 * K - 5) / (T - K - 3))
              * ((T - 2 * K - 3) / (T - 2 * K - 1) * Wb - K))
    p = lambda z: float(2 * (1 - stats.norm.cdf(abs(z))))  # noqa: E731
    return DHResult(cause, effect, K, Wb, float(Zb), p(Zb), float(Zt),
                    p(Zt) if np.isfinite(Zt) else np.nan, N, T_all, ind)
