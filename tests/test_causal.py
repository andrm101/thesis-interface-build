"""Recovery tests for causal.py: plant a known effect, check the estimator
finds it."""
import numpy as np
import pandas as pd
import pytest

import causal as C


def _panel(n=16, T=26, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1998, 1998 + T)
    common = np.cumsum(rng.normal(0.02, 0.01, T))       # shared cycle
    rows = []
    for i in range(n):
        a = rng.normal(10, 0.3)
        for k, yr in enumerate(years):
            rows.append(dict(Country=f"C{i:02d}", Year=yr,
                             logy=a + common[k] + 0.005 * rng.normal()))
    return pd.DataFrame(rows)


def test_parse_events():
    assert C.parse_events("Poland:2016; Czechia:2005, bad") == \
        {"Poland": 2016, "Czechia": 2005}


def test_event_study_recovers_planted_att():
    d = _panel()
    events = {"C00": 2005, "C01": 2005, "C02": 2005, "C03": 2010, "C04": 2010}
    for c, g in events.items():
        m = (d.Country == c) & (d.Year >= g)
        d.loc[m, "logy"] += 0.05                       # +5 % from event on
    d["Y"] = np.exp(d["logy"])
    r = C.event_study(d, "Y", events, pre=3, post=5, n_boot=99)
    post = r.by_event_time.query("e >= 0")["att"]
    pre = r.by_event_time.query("e < -1")["att"]
    assert np.allclose(post, 5, atol=0.6)
    assert np.allclose(pre, 0, atol=0.6)
    assert r.pretrend_p > 0.05


def test_not_yet_treated_and_detrend_remove_linear_pretrend():
    d = _panel()
    events = {f"C0{i}": 2008 for i in range(4)} | {"C04": 2014, "C05": 2014}
    for c, g in events.items():                    # treated diverge early
        m = d.Country == c
        d.loc[m, "logy"] += 0.01 * (d.loc[m, "Year"] - 1998)
    d["Y"] = np.exp(d["logy"])
    raw = C.event_study(d, "Y", events, 4, 4, n_boot=49)
    adj = C.event_study(d, "Y", events, 4, 4, n_boot=49, detrend=True)
    assert raw.overall_post > 1.5                  # spurious effect
    assert abs(adj.overall_post) < 0.5             # removed by detrending
    assert raw.pretrend_p < 0.05
    ny = C.event_study(d, "Y", events, 4, 4, n_boot=49,
                       control=C.CONTROL_GROUPS[1])
    assert set(ny.treated) == set(events)


def test_synthetic_control_recovers_weights_and_gap():
    d = _panel(n=8, seed=3)
    w = pd.pivot_table(d, index="Year", columns="Country", values="logy")
    rng = np.random.default_rng(1)
    donors = w.columns[1:]
    # C00 := 0.6·C01 + 0.4·C02 path, +10 % after 2012
    path = 0.6 * w["C01"] + 0.4 * w["C02"] + rng.normal(0, 1e-4, len(w))
    path[w.index >= 2012] += 0.10
    d.loc[d.Country == "C00", "logy"] = path.values
    d["Y"] = np.exp(d["logy"])
    r = C.synthetic_control(d, "Y", "C00", 2012, list(donors), demean=False)
    assert r.weights[["C01", "C02"]].sum() > 0.95
    assert abs(r.avg_post_gap - 10) < 1
    assert r.p_value <= 1 / len(donors) + 1e-9     # most extreme ratio


def test_local_projections_recover_dynamic_response():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        x, ly, shocks = 1.0, 10.0, []
        for t in range(30):
            dx = rng.normal(0, 0.3)
            x += dx
            shocks.append(dx)
            # y responds with 2 % per unit shock, from one year later on
            ly += 0.02 * (shocks[-2] if len(shocks) > 1 else 0) \
                + rng.normal(0, 0.002)
            rows.append(dict(Country=f"C{i}", Year=1990 + t, X=x,
                             Y=np.exp(ly)))
    d = pd.DataFrame(rows)
    lp = C.local_projections(d, "Y", "X", horizons=3, n_lags=1)
    b = lp.set_index("h")["beta"]
    assert abs(b[0]) < 0.5 and abs(b[2] - 2) < 0.5


def test_dml_recovers_theta_and_heterogeneity():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        ly = 10.0
        for t in range(25):
            z = rng.normal()
            gap = 1 + 0.5 * rng.normal()
            D = 0.5 * z + rng.normal()
            rows.append(dict(Country=f"C{i}", Year=2000 + t, D=D, Z=z,
                             Gap=gap))
    d = pd.DataFrame(rows)
    # growth_t = (1 − 0.8·(gap_{t−1}−1))·D_{t−1} + sin(Z_{t−1}) + noise
    g = d.groupby("Country")
    Dl, Zl, Gl = g.D.shift(1), g.Z.shift(1), g.Gap.shift(1)
    growth = ((1 - 0.8 * (Gl - 1)) * Dl + np.sin(Zl)
              + 0.1 * rng.normal(size=len(d))).fillna(0)
    d["Y"] = np.exp(10 + g["D"].transform(lambda s: 0) +
                    growth.groupby(d.Country).cumsum() / 100)
    r = C.dml_plr(d, "Y", "D", ["Z"], moderator="Gap",
                  learner="Random Forest", n_folds=4)
    assert abs(r.theta - 1) < 0.25
    assert r.cate_slope < -0.4
