"""Tests for CIPS / Dumitrescu-Hurlin (panel_tests.py), held-out importance
(ml_eval.py) and the causal scenario (causal.py)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import causal as C
import ml_eval
import panel_tests as P


def _long(Y, name="v", start=1990):
    T, N = Y.shape
    return pd.DataFrame([dict(Country=f"C{i:02d}", Year=start + t,
                              **{name: Y[t, i]})
                         for i in range(N) for t in range(T)])


def test_cips_rejects_stationary_not_random_walk():
    rng = np.random.default_rng(0)
    T, N = 40, 15
    f = rng.normal(size=T)
    stat = np.zeros((T, N))
    for t in range(1, T):
        stat[t] = 0.3 * stat[t - 1] + f[t] + rng.normal(size=N)
    rw = np.cumsum(rng.normal(size=(T, N)), axis=0) + np.cumsum(f)[:, None]
    assert P.cips(_long(stat), "v", 1, n_sim=200).reject
    assert not P.cips(_long(rw), "v", 1, n_sim=200).reject


def test_dumitrescu_hurlin_detects_planted_direction():
    rng = np.random.default_rng(1)
    T, N = 30, 15
    x = rng.normal(size=(T, N))
    y = np.zeros((T, N))
    for t in range(1, T):
        y[t] = 0.3 * y[t - 1] + 0.8 * x[t - 1] + rng.normal(size=N)
    df = _long(x, "x").merge(_long(y, "y"), on=["Country", "Year"])
    fwd = P.dumitrescu_hurlin(df, "x", "y", 1)
    back = P.dumitrescu_hurlin(df, "y", "x", 1)
    assert fwd.p_Z_tilde < 0.001
    assert back.p_Z_tilde > 0.05


def test_grouped_importance_ranks_signal_over_noise():
    rng = np.random.default_rng(2)
    n_c, T = 12, 15
    g = np.repeat(np.arange(n_c), T)
    X = pd.DataFrame({"signal": rng.normal(size=n_c * T),
                      "noise": rng.normal(size=n_c * T)})
    y = pd.Series(2 * X.signal + 0.3 * rng.normal(size=n_c * T))
    t = ml_eval.grouped_permutation_importance(
        Ridge(), X, y, g, np.tile(np.arange(T), n_c), n_repeats=5)
    assert t.iloc[0].feature == "signal" and t.iloc[0]["mean"] > 1
    assert abs(t.set_index("feature").loc["noise", "mean"]) < 0.05


def test_causal_scenario_arithmetic():
    df = pd.DataFrame({"Country": "A", "Year": range(2000, 2011),
                       "Y": np.exp(np.linspace(10, 10.1, 11))})
    lp = pd.DataFrame({"h": range(4), "state": "S", "beta": [0, 1, 2, 3],
                       "lo": [0, 0, 0, 0], "hi": [0, 2, 4, 6]})

    class _D:                         # minimal DMLResult stand-in
        def effect_at(self, m):
            return 0.5, 0.1
        moderator_mean = 0.0

    out = C.causal_scenario(df, "Y", "A", 2.0, 3, lp, "S", _D(), 0.0)
    base = np.log(out.baseline) * 100
    np.testing.assert_allclose(np.log(out.lp_mid) * 100 - base,
                               [0, 2, 4, 6], atol=1e-9)
    np.testing.assert_allclose(np.log(out.dml_mid) * 100 - base,
                               [0, 1, 2, 3], atol=1e-9)
    assert abs(out.attrs["baseline_growth_pct"] - 1.0) < 1e-9
