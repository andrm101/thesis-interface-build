"""Tests for the z-score screen (outliers.py) and EDA test battery (eda.py)."""
import numpy as np
import pandas as pd
import pytest

import eda
import outliers
from conftest import DATA_PATH


@pytest.fixture(scope="module")
def raw():
    return pd.read_excel(DATA_PATH)


def _toy():
    rng = np.random.default_rng(0)
    # Uniform noise: |z| can never exceed √3, so only planted outliers flag.
    rows = [(c, y, 10 + rng.uniform(-1, 1), 5 + rng.uniform(-1, 1))
            for c in "ABCDEFGH" for y in range(2000, 2020)]
    df = pd.DataFrame(rows, columns=["Country", "Year", "x", "y"])
    df.loc[df.Country == "H", "x"] += 30          # outlying country
    df.loc[(df.Country == "A") & (df.Year == 2010), "y"] = 500  # data spike
    return df


def test_default_screen_drops_only_luxembourg(raw):
    res = outliers.screen(raw)
    assert list(res.flagged_countries) == ["Luxembourg"]
    assert "Luxembourg" not in set(res.df["Country"])
    assert len(res.df) == len(raw) - 24


def test_country_and_observation_levels():
    res = outliers.screen(_toy(), cols=["x", "y"], level="Both",
                          threshold=2.5)
    assert list(res.flagged_countries) == ["H"]
    spike = res.flagged_obs.query("Variable == 'y'")
    assert (spike.Country == "A").all() and (spike.Year == 2010).all()
    assert res.df.loc[(res.df.Country == "A") & (res.df.Year == 2010),
                      "y"].isna().all()


@pytest.mark.parametrize("action,rows,nans", [
    ("Set to NaN", 160, 1), ("Winsorize", 160, 0), ("Drop row", 159, 0)])
def test_observation_actions(action, rows, nans):
    res = outliers.screen(_toy(), cols=["y"], level="Observations",
                          obs_action=action, threshold=3.0)
    assert len(res.df) == rows
    assert res.df["y"].isna().sum() == nans
    assert res.df["y"].max() < 500


def test_spike_screened_before_country_means(raw):
    # Italy 2011 'Labor' typo must be caught as a cell, not drop Italy.
    cols = eda.numeric_vars(raw)
    res = outliers.screen(raw, cols=cols, level="Both")
    assert "Italy" not in res.flagged_countries
    fo = res.flagged_obs
    assert ((fo.Country == "Italy") & (fo.Year == 2011)
            & (fo.Variable == "Labor")).any()


def test_bh_adjust_matches_definition():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    adj = eda.bh_adjust(p)
    assert np.isnan(adj[3])
    # sorted p = .01,.03,.04,.5 (m=4): raw p·m/rank = .04,.06,.0533,.5;
    # step-up running min → .04,.0533,.0533,.5
    np.testing.assert_allclose(adj[[0, 2, 1, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert (adj[np.isfinite(adj)] >= p[np.isfinite(p)]).all()


def test_eda_battery_runs(raw):
    df = outliers.screen(raw).df
    groups = pd.Series({c: ("A" if i % 2 else "B")
                        for i, c in enumerate(df.Country.unique())})
    for out in (eda.normality_tests(df), eda.group_difference_tests(df, groups),
                eda.country_effect_tests(df), eda.trend_tests(df),
                eda.correlation_tests(df, "Y by L"), eda.cd_tests(df),
                eda.variance_decomposition(df), eda.structural_break_tests(df)):
        assert len(out) > 0
    ll = eda.lead_lag_correlation(df, "PIB towards research", "Y by L", 3)
    assert list(ll.Lag) == [-3, -2, -1, 0, 1, 2, 3]
    vd = eda.variance_decomposition(df)
    np.testing.assert_allclose(vd.Between_share + vd.Within_share, 1)


def test_pesaran_cd_detects_common_factor():
    rng = np.random.default_rng(1)
    f = rng.normal(size=30)
    rows = [(c, t, f[t] + 0.3 * rng.normal()) for c in range(10) for t in range(30)]
    df = pd.DataFrame(rows, columns=["Country", "Year", "v"])
    cd, p, *_ = eda.pesaran_cd(df, "v")
    assert cd > 10 and p < 1e-6


def test_chow_detects_break():
    t = np.arange(2000, 2024, dtype=float)
    y = np.where(t < 2009, t - 2000, 20 - 2 * (t - 2009))
    F, p = eda.chow_test(t, y + 0.01 * np.sin(t), 2009)
    assert p < 1e-6
