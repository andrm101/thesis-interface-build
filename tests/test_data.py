"""Checks on the shipped dataset and the pure-Python helpers."""
import pandas as pd
import pytest

from conftest import DATA_PATH
from constants import EU25, INNOVATIVE_CLUSTER, EMERGING_CLUSTER, stars, detect_variable_role

REQUIRED = ["Country", "Year", "Y by L", "Savings Percentage", "Human Capital Proxy",
            "Labor in research", "Labor not in research", "PIB towards research",
            "Patents per capita"]


@pytest.fixture(scope="module")
def df():
    return pd.read_excel(DATA_PATH)


def test_schema(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"panel_data.xlsx missing columns: {missing}"


def test_documented_coverage(df):
    # README / app header advertise 23 countries x 2000-2023, balanced.
    assert df["Country"].nunique() == 23
    assert (df["Year"].min(), df["Year"].max()) == (2000, 2023)
    assert len(df) == 552
    assert not df.duplicated(["Country", "Year"]).any()


def test_countries_are_in_thesis_clusters(df):
    names = {c.lower() for c in df["Country"].unique()}
    assert names <= set(EU25), f"unclustered countries: {names - set(EU25)}"


def test_clusters_disjoint():
    assert not set(INNOVATIVE_CLUSTER) & set(EMERGING_CLUSTER)


@pytest.mark.parametrize("p,expected", [(0.001, "***"), (0.03, "**"), (0.07, "*"), (0.5, "")])
def test_stars(p, expected):
    assert stars(p) == expected


def test_detect_variable_role():
    assert detect_variable_role("Y by L") == "output_per_worker"
    assert detect_variable_role("Savings Percentage") == "savings"
    assert detect_variable_role("Patents per capita") == "patents"
