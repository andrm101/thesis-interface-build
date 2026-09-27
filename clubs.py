"""
clubs.py — Phillips & Sul (2007, 2009) log-t convergence test and club
clustering algorithm, with the Schnurbus, Haupt & Meier (2017) club-merging
step.

Model: log X_it = δ_it μ_t. Countries converge if the relative transition
parameters h_it = log X_it / mean_i(log X_it) all tend to 1. With
H_t = mean_i (h_it − 1)², the log-t regression

    log(H_1 / H_t) − 2·log(log t) = a + b·log t + u_t,   t = [rT], …, T

is run with HAC standard errors; convergence is rejected when the one-sided
t-statistic on b is below −1.65. b ≥ 0 indicates convergence; 2·b estimates
the speed of convergence (b ≥ 2: convergence in levels).

Input levels must be positive. For year-on-year growth indices
(previous year = 100) use `chain_index`, which builds a level path with
every country starting at 100 — clubs then describe *cumulative growth
paths since the base year*, not convergence of absolute productivity
levels (which needs level data such as PWT output per worker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

CRIT = -1.65


# ── Data preparation ───────────────────────────────────────────────────────
def chain_index(df: pd.DataFrame, col: str, base: float = 100.0) -> pd.Series:
    """Cumulate a growth index (prev. year = base) into a level path
    starting at 1 in each country's first year."""
    d = df.sort_values(["Country", "Year"])
    g = (d[col] / base).groupby(d["Country"])
    lvl = g.cumprod() / g.transform("first")
    return lvl.reindex(df.index)


def positive_log_scale(wide: pd.DataFrame) -> float:
    """Smallest power of 10 that puts every value above e (log > 1).

    h_it divides by the cross-country mean of *log* levels, so logs near
    or below zero (shares such as 0.02, or an index at its base of 1)
    make the transition paths explode or flip sign. Rescaling is the usual
    remedy (e.g. index base = 100, shares in per-mille).
    """
    lo = float(np.nanmin(wide.to_numpy()))
    return float(10 ** max(0, int(np.ceil(np.log10(np.e / lo))))) if lo > 0 else 1.0


def prepare_panel(df: pd.DataFrame, col: str, chain: bool,
                  hp_lambda: float | None = 400.0) -> pd.DataFrame:
    """Wide (Year × Country) panel of log levels, HP-filtered trend.

    Chained indices are based at 100; other series are rescaled by
    `positive_log_scale`. Countries with missing or non-positive values are dropped.
    Phillips & Sul recommend λ = 400 for annual data.
    """
    s = chain_index(df, col) * 100 if chain else df[col]
    wide = (df.assign(_v=s).pivot_table(index="Year", columns="Country",
                                        values="_v"))
    wide = wide.loc[:, wide.notna().all() & (wide > 0).all()]
    wide = wide * positive_log_scale(wide)
    logw = np.log(wide)
    if hp_lambda:
        logw = logw.apply(lambda c: hpfilter(c, lamb=hp_lambda)[1])
    return logw


# ── log-t test ─────────────────────────────────────────────────────────────
@dataclass
class LogT:
    b: float
    se: float
    t: float
    n: int
    converges: bool


def log_t(logx: pd.DataFrame, r: float = 0.3) -> LogT:
    """Phillips-Sul log-t test on a wide Year × Country frame of log levels."""
    X = logx.to_numpy(float)
    T, N = X.shape
    cross_mean = X.mean(axis=1, keepdims=True)
    h = X / cross_mean
    H = ((h - 1) ** 2).mean(axis=1)
    start = max(int(np.floor(r * T)), 1)     # drop first r·T periods
    t_idx = np.arange(1, T + 1)[start:]
    Ht = H[start:]
    if (Ht <= 0).any() or len(t_idx) < 4:
        return LogT(np.nan, np.nan, np.nan, N, False)
    yv = np.log(H[0] / Ht) - 2 * np.log(np.log(t_idx))
    Xr = sm.add_constant(np.log(t_idx))
    lags = int(np.floor(4 * (len(t_idx) / 100) ** (2 / 9)))
    fit = sm.OLS(yv, Xr).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    b, se = fit.params[1], fit.bse[1]
    tval = b / se
    return LogT(float(b), float(se), float(tval), N, bool(tval > CRIT))


# ── Club clustering ────────────────────────────────────────────────────────
@dataclass
class ClubResult:
    full: LogT
    clubs: list[list[str]] = field(default_factory=list)
    club_tests: list[LogT] = field(default_factory=list)
    divergent: list[str] = field(default_factory=list)
    initial_clubs: list[list[str]] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)

    def membership(self) -> pd.DataFrame:
        rows = [(c, k + 1) for k, cl in enumerate(self.clubs) for c in cl]
        rows += [(c, 0) for c in self.divergent]
        return pd.DataFrame(rows, columns=["Country", "Club"])


def _core(logx: pd.DataFrame, order: list[str], r: float):
    """Largest-t core group among the top-ranked countries (Step 2).

    Scans k = 2, 3, … over the ordering; starts at the first k whose
    t_k > −1.65 and grows while it keeps passing, picking the k with the
    maximum t. Returns (core members, t) or (None, None).
    """
    n = len(order)
    for start in range(n - 1):
        ts = []
        for k in range(start + 2, n + 1):
            res = log_t(logx[order[start:k]], r)
            if res.t > CRIT:
                ts.append((res.t, k))
            elif ts:
                break
        if ts:
            _, kbest = max(ts)
            return order[start:kbest], start
    return None, None


def _find_clubs(logx: pd.DataFrame, order: list[str], r: float,
                c_star: float) -> tuple[list[list[str]], list[str]]:
    clubs: list[list[str]] = []
    remaining = list(order)
    while len(remaining) >= 2:
        whole = log_t(logx[remaining], r)
        if whole.converges:
            clubs.append(remaining)
            return clubs, []
        core, start = _core(logx, remaining, r)
        if core is None:
            return clubs, remaining          # no convergent pair left
        # Step 3: sieve the others one at a time against the core.
        club = list(core)
        for c in remaining:
            if c in core:
                continue
            if log_t(logx[core + [c]], r).t > c_star:
                club.append(c)
        # Guard: the full club must itself pass; if not, keep the core.
        if not log_t(logx[club], r).converges:
            club = list(core)
        clubs.append([c for c in order if c in club])
        remaining = [c for c in remaining if c not in club]
    return clubs, remaining


def club_clustering(logx: pd.DataFrame, r: float = 0.3,
                    c_star: float = 0.0, merge: bool = True) -> ClubResult:
    """Run the full Phillips-Sul procedure on a Year × Country log panel.

    Countries are ordered by the average of the last third of the sample.
    *c_star* = 0 is the conservative sieve criterion Phillips & Sul
    recommend for short T.
    """
    tail = logx.iloc[-max(1, len(logx) // 3):].mean()
    order = tail.sort_values(ascending=False).index.tolist()
    res = ClubResult(full=log_t(logx, r))
    clubs, divergent = _find_clubs(logx, order, r, c_star)
    res.initial_clubs = [list(c) for c in clubs]

    if merge:   # Schnurbus et al. (2017): merge adjacent clubs if they pass
        i = 0
        while i < len(clubs) - 1:
            merged = clubs[i] + clubs[i + 1]
            test = log_t(logx[merged], r)
            if test.converges:
                res.merges.append(f"Club {i + 1} + Club {i + 2} "
                                  f"(t = {test.t:.2f})")
                clubs[i] = merged
                del clubs[i + 1]
            else:
                i += 1

    res.clubs = clubs
    res.club_tests = [log_t(logx[c], r) for c in clubs]
    res.divergent = divergent
    return res


def transition_paths(logx: pd.DataFrame) -> pd.DataFrame:
    """Relative transition paths h_it (Year × Country)."""
    return logx.div(logx.mean(axis=1), axis=0)
