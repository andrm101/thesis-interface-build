"""Tests for the raw-workbook importer (build_panel.py)."""
import os

import numpy as np
import pandas as pd
import pytest

import build_panel as bp

pytestmark = pytest.mark.skipif(not os.path.exists(bp.DEFAULT_RAW),
                                reason="raw workbook not present")


@pytest.fixture(scope="module")
def panel():
    return bp.build()


@pytest.fixture(scope="module")
def thesis():
    p = pd.read_excel(os.path.join(os.path.dirname(bp.__file__),
                                   "panel_data.xlsx"))
    p["Country"] = p["Country"].replace({"Slovak Republic": "Slovakia"})
    return p


def test_shape_and_coverage(panel):
    assert panel["Country"].nunique() == 26
    assert (panel["Year"].min(), panel["Year"].max()) == (1998, 2023)
    assert not panel.duplicated(["Country", "Year"]).any()


def test_tertiary_share_uses_correctly_aligned_source(panel):
    de = panel.query("Country == 'Germany' and Year == 2000")
    it = panel.query("Country == 'Italy' and Year == 2000")
    # Germany ≈ 21 % tertiary in 2000, Italy ≈ 8 % (the AAA block swapped
    # Italy's series into Germany's row).
    assert de["Tertiary_share"].item() > 0.18
    assert it["Tertiary_share"].item() < 0.12


def test_researcher_unit_fix(panel):
    lir = panel.groupby("Country")["Labor in research"].mean()
    assert lir[["Czechia", "Estonia"]].max() < 0.03


@pytest.mark.parametrize("col", ["Y by L", "PIB towards research",
                                 "Patents per capita"])
def test_reproduces_thesis_growth_indices(panel, thesis, col):
    m = thesis.merge(panel, on=["Country", "Year"], suffixes=("_t", "_r"))
    np.testing.assert_allclose(m[f"{col}_t"], m[f"{col}_r"], rtol=1e-9)


def test_level_rd_clusters_match_thesis_typology(panel):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from constants import INNOVATIVE_CLUSTER
    cols = ["RD_pct_GDP", "Researchers_per_1000_emp",
            "Patents_per_million", "Tertiary_share"]
    cm = (panel[panel.Year >= 2000].groupby("Country")[cols].mean()
          .dropna().drop(index="Luxembourg"))
    lab = KMeans(2, random_state=42, n_init=10).fit_predict(
        StandardScaler().fit_transform(cm))
    inno = set(cm.index[lab == lab[list(cm.index).index("Germany")]])
    assert inno == {c for c in cm.index if c.lower() in INNOVATIVE_CLUSTER}
