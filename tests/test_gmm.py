"""Recovery tests for gmm.py and the knowledge stock in build_panel.py.

Reference values were cross-checked against pydynpd (xtabond2 port):
difference GMM coefficients, Windmeijer SEs, Hansen J and AR tests agree.
"""
import numpy as np
import pandas as pd
import pytest

import build_panel
import gmm


def _dgp(rho=(0.5,), beta=1.0, N=60, T=12, seed=0):
    """y_it = Σρ_j y_{t−j} + β x_it + η_i + ε_it, x endogenous (∝ ε_it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        eta = rng.normal()
        ys = [eta / (1 - sum(rho))] * len(rho)
        xp = 0.0
        for t in range(T + 20):
            e = rng.normal()
            x = 0.5 * xp + 0.5 * eta + 0.6 * e + rng.normal()
            y = sum(r * ys[-1 - j] for j, r in enumerate(rho)) + beta * x \
                + eta + e
            ys.append(y)
            xp = x
            if t >= 20:
                rows.append(dict(Country=i, Year=t, y=y, x=x))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["difference", "system"])
def test_recovers_planted_parameters(method):
    d = _dgp()
    r = gmm.estimate(d, "y", endog=["x"], method=method, lags=(2, 4),
                     time_demean=False)
    assert abs(r.params["L1.y"] - 0.5) < 0.12
    assert abs(r.params["x"] - 1.0) < 0.25
    assert r.ar1_p < 0.01 and r.ar2_p > 0.05 and r.hansen_p > 0.05
    assert isinstance(r.pvalues, pd.Series)


def test_matches_pydynpd_reference():
    # Two-step difference GMM on this DGP, pydynpd 0.2.2:
    # L1.y 0.4383682 (SE 0.0633346), x 1.0236768 (SE 0.1675283),
    # Hansen 4.176, AR(1) z −4.67, AR(2) z 1.14.
    r = gmm.estimate(_dgp(), "y", endog=["x"], method="difference",
                     lags=(2, 4), time_demean=False)
    np.testing.assert_allclose(r.params.values, [0.4383682, 1.0236768],
                               atol=1e-6)
    np.testing.assert_allclose(r.se.values, [0.0633346, 0.1675283],
                               atol=1e-6)
    assert abs(r.hansen - 4.176) < 1e-3
    assert abs(r.ar1 + 4.67) < 0.01 and abs(r.ar2 - 1.14) < 0.01


def test_gmm_beats_fixed_effects_under_endogeneity():
    from linearmodels.panel import PanelOLS
    d = _dgp().sort_values(["Country", "Year"])
    d["y_l1"] = d.groupby("Country")["y"].shift(1)
    dd = d.dropna().set_index(["Country", "Year"])
    fe = PanelOLS(dd.y, dd[["y_l1", "x"]], entity_effects=True).fit()
    g = gmm.estimate(d, "y", endog=["x"], lags=(2, 4), time_demean=False)
    assert abs(fe.params["x"] - 1) > 0.3            # FE biased
    assert abs(g.params["x"] - 1) < abs(fe.params["x"] - 1) / 2


def test_two_lag_dynamics_and_long_run():
    d = _dgp(rho=(0.4, 0.2), N=80, seed=3)
    r = gmm.estimate(d, "y", endog=["x"], lags=(2, 4), y_lags=2,
                     time_demean=False)
    assert abs(r.params["L1.y"] + r.params["L2.y"] - 0.6) < 0.15
    lr, se = r.long_run["x"]
    assert abs(lr - 1.0 / 0.4) < 3 * se


def test_knowledge_stock_perpetual_inventory():
    flow = pd.Series([10.0] * 8)                    # constant flow, g = 0
    k = build_panel.knowledge_stock(flow, delta=0.15)
    np.testing.assert_allclose(k, 10 / 0.15)        # steady state R/δ
    growing = pd.Series(10 * 1.05 ** np.arange(8))
    kg = build_panel.knowledge_stock(growing, delta=0.15)
    assert abs(kg.iloc[0] - 10 / (0.05 + 0.15)) < 1e-9
    np.testing.assert_allclose(kg.iloc[1], 0.85 * kg.iloc[0] + growing.iloc[1])
