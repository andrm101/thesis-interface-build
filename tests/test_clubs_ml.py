"""Tests for Phillips-Sul clubs (clubs.py) and leakage-safe CV (ml_eval.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import clubs
import ml_eval
import outliers
from conftest import DATA_PATH


def _two_club_panel(T=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(1, T + 1)
    cols = {}
    for tag, lvl in (("H", 200), ("L", 50)):
        for i in range(6):
            cols[f"{tag}{i}"] = (np.log(lvl) + rng.uniform(-.5, .5)
                                 * np.exp(-0.15 * t) + 0.02 * t)
    return pd.DataFrame(cols, index=2000 + t)


def test_logt_accepts_convergent_and_rejects_mixed():
    lx = _two_club_panel()
    assert clubs.log_t(lx[[f"H{i}" for i in range(6)]]).converges
    assert not clubs.log_t(lx).converges


def test_club_clustering_recovers_planted_clubs():
    res = clubs.club_clustering(_two_club_panel())
    assert [sorted(c) for c in res.clubs] == [
        [f"H{i}" for i in range(6)], [f"L{i}" for i in range(6)]]
    assert res.divergent == []
    mem = res.membership()
    assert set(mem.Club) == {1, 2}


def test_chain_index_and_positive_logs():
    df = pd.DataFrame({"Country": ["A"] * 3 + ["B"] * 3,
                       "Year": [1, 2, 3] * 2,
                       "g": [110, 110, 90, 100, 100, 100]})
    lvl = clubs.chain_index(df, "g")
    np.testing.assert_allclose(lvl, [1, 1.1, 0.99, 1, 1, 1])
    wide = pd.DataFrame({"x": [0.002, 0.03]})
    assert np.log(wide * clubs.positive_log_scale(wide)).min().min() > 1


def test_real_data_clubs_run():
    df = outliers.screen(pd.read_excel(DATA_PATH)).df
    lx = clubs.prepare_panel(df, "Y by L", chain=True)
    res = clubs.club_clustering(lx)
    placed = sum(len(c) for c in res.clubs) + len(res.divergent)
    assert placed == lx.shape[1]
    assert all(t.converges for t in res.club_tests)


def _panel(n_c=10, T=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_c):
        fe = rng.normal(scale=5)          # country level y depends on
        for t in range(T):
            x = fe + rng.normal()
            rows.append((f"C{c}", 2000 + t, x, fe + 0.1 * rng.normal()))
    d = pd.DataFrame(rows, columns=["Country", "Year", "x", "y"])
    return d[["x"]], d["y"], d["Country"], d["Year"]


def test_grouped_splits_never_share_countries():
    X, y, g, yr = _panel()
    for tr, te in ml_eval.cv_splits(ml_eval.SCHEMES[0], g, yr, 5, len(X)):
        assert not set(g.iloc[tr]) & set(g.iloc[te])
    tr, te = ml_eval.holdout_split(ml_eval.SCHEMES[0], g, yr, len(X))
    assert not set(g.iloc[tr]) & set(g.iloc[te])


def test_time_splits_train_strictly_before_test():
    X, y, g, yr = _panel()
    splits = ml_eval.cv_splits(ml_eval.SCHEMES[1], g, yr, 4, len(X))
    assert len(splits) >= 2
    for tr, te in splits:
        assert yr.iloc[tr].max() < yr.iloc[te].min()


@pytest.mark.parametrize("scheme", ml_eval.SCHEMES)
def test_evaluate_returns_consistent_metrics(scheme):
    X, y, g, yr = _panel()
    r = ml_eval.evaluate(Ridge(), X, y, g, yr, scheme, 5)
    for k in ("cv_r2_mean", "test_r2", "random_cv_r2", "cv_rmse_mean"):
        assert np.isfinite(r[k])
    assert len(r["y_te"]) == len(r["y_pred"])
    assert r["scaler"].transform(X).shape == X.shape
