"""Web API (api/main.py): every endpoint returns well-formed JSON built from
the same analysis functions reproduce.py uses."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api import service as S
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_meta(client):
    m = client.get("/api/meta").json()
    names = {c["name"] for c in m["countries"]}
    assert "Luxembourg" not in names and len(names) == 24      # z-screen
    assert {"EU accession"} <= set(m["event_sets"])
    assert any(v["key"] == "RD_pct_GDP" and v["shock"] for v in m["variables"])


def test_series_and_snapshot(client):
    r = client.get("/api/series", params={"variable": "RD_pct_GDP",
                                          "countries": "Poland,Sweden"}).json()
    assert [s["country"] for s in r["series"]] == ["Poland", "Sweden"]
    assert len(r["series"][0]["values"]) == len(r["years"])
    snap = client.get("/api/snapshot", params={"variable": "RD_pct_GDP",
                                               "year": 2020}).json()
    vals = [x["value"] for x in snap["rows"]]
    assert vals == sorted(vals, reverse=True)
    assert client.get("/api/series", params={"variable": "nope"}).status_code == 404


def test_typology_matches_reproduce(client):
    t = client.get("/api/typology").json()
    km = S.R.kmeans_typology(S.df(), S.R.LEVEL_RD)
    inno = {p["country"] for p in t["points"] if p["cluster"] == "Innovative"}
    assert inno == set(km["innovative"])
    assert abs(t["silhouette"] - km["silhouette"]) < 1e-6


def test_clubs(client):
    c = client.get("/api/clubs").json()
    members = [m for k in c["clubs"] for m in k["members"]] + c["divergent"]
    assert sorted(members) == sorted(p["country"] for p in c["paths"])


def test_local_projections_levels_split(client):
    r = client.post("/api/local-projections",
                    json={"outcome": "RD_pct_GDP", "shock": "GBARD_pct_GDP",
                          "horizons": 3, "levels": True, "split": True}).json()
    assert {x["state"] for x in r["rows"]} == {"Emerging", "Innovative"}
    assert len(r["rows"]) == 8
    bad = client.post("/api/local-projections",
                      json={"outcome": "RD_pct_GDP", "shock": "x"})
    assert bad.status_code == 404


def test_event_study_custom(client):
    r = client.post("/api/event-study",
                    json={"outcome": "RD_pct_GDP", "event_set": "Custom",
                          "custom": "Poland:2010, Czechia:2008", "n_boot": 19})
    assert r.status_code == 200
    body = r.json()
    assert set(body["events"]) == {"Poland", "Czechia"}
    assert body["by_event_time"][0]["e"] == -3
    empty = client.post("/api/event-study",
                        json={"outcome": "RD_pct_GDP", "event_set": "Custom",
                              "custom": "", "n_boot": 19})
    assert empty.status_code == 422


def test_results_and_coverage(client):
    idx = client.get("/api/results").json()
    assert {"A1", "B13", "D5"} <= {s["id"] for s in idx}
    md = client.get("/api/results/D5").json()["markdown"]
    assert md.startswith("**D5 ·") and "|" in md
    cov = client.get("/api/coverage").json()
    assert len(cov["cells"]) == len(cov["countries"]) * len(cov["variables"])
