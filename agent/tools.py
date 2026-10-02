"""
tools.py — what the assistant may run. Each tool wraps a function of
api/service.py (the same analysis layer as the dashboard and reproduce.py),
is read-only, and has a strict JSON schema whose enums list the real
variables, countries and event sets, so the model cannot ask for something
that does not exist.
"""
from __future__ import annotations

from functools import lru_cache

from api import service as S

N_BOOT = 99             # event-study bootstrap draws (≈ 9 s)


def _r(x, d=4):
    return None if x is None else round(float(x), d)


@lru_cache(maxsize=1)
def _enums() -> dict:
    m = S.meta()
    return {
        "variables": [v["key"] for v in m["variables"]],
        "shocks": [v["key"] for v in m["variables"] if v["shock"]],
        "countries": [c["name"] for c in m["countries"]],
        "event_sets": list(m["event_sets"]) + ["Custom"],
        "controls": m["control_groups"],
        "sections": [s["id"] for s in S.results_index()],
    }


def _schema(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def definitions() -> list[dict]:
    e = _enums()
    var = {"type": "string", "enum": e["variables"]}
    return [
        {"name": "list_variables", "strict": True,
         "description": "Variables (with units and coverage), countries with their "
                        "Innovative/Emerging group, the year range and the available "
                        "event sets. Call first if unsure what exists.",
         "input_schema": _schema({})},
        {"name": "describe_variable", "strict": True,
         "description": "First and last value, change and mean of a variable for "
                        "the given countries (empty list = all), plus group averages "
                        "in the last year.",
         "input_schema": _schema({
             "variable": var,
             "countries": {"type": "array",
                           "items": {"type": "string", "enum": e["countries"]}}})},
        {"name": "rank_countries", "strict": True,
         "description": "All countries ranked by a variable in one year.",
         "input_schema": _schema({"variable": var, "year": {"type": "integer"}})},
        {"name": "local_projection", "strict": True,
         "description": "Panel local projections (country and year fixed effects, "
                        "clustered SE): response of `outcome` h = 0..horizons years "
                        "after a one-unit rise in `shock`. levels=false gives % "
                        "responses, levels=true responses in the outcome's units. "
                        "split_by_group estimates Emerging and Innovative separately.",
         "input_schema": _schema({
             "outcome": var,
             "shock": {"type": "string", "enum": e["shocks"]},
             "horizons": {"type": "integer"},
             "levels": {"type": "boolean"},
             "split_by_group": {"type": "boolean"}})},
        {"name": "event_study", "strict": True,
         "description": "Callaway-Sant'Anna staggered event study: % effect on "
                        "`outcome` by years since the event, the average post-event "
                        "effect and a pre-trend test. For event_set 'Custom', give "
                        "events as 'Country:Year, Country:Year'; otherwise pass ''.",
         "input_schema": _schema({
             "outcome": var,
             "event_set": {"type": "string", "enum": e["event_sets"]},
             "custom_events": {"type": "string"},
             "control": {"type": "string", "enum": e["controls"]},
             "detrend": {"type": "boolean"}})},
        {"name": "convergence_clubs", "strict": True,
         "description": "Phillips-Sul convergence clubs in log output per worker.",
         "input_schema": _schema({})},
        {"name": "typology", "strict": True,
         "description": "K-Means R&D typology (Innovative vs Emerging) with "
                        "silhouette and agreement with the thesis grouping.",
         "input_schema": _schema({})},
        {"name": "result_section", "strict": True,
         "description": "The text and tables of one section of results/RESULTS.md "
                        "(the thesis's reproduced headline results), e.g. 'D5'.",
         "input_schema": _schema({
             "section_id": {"type": "string", "enum": e["sections"]}})},
    ]


# ── implementations ────────────────────────────────────────────────────────
def list_variables() -> dict:
    m = S.meta()
    return {"years": m["years"], "variables": m["variables"],
            "countries": m["countries"],
            "event_sets": {k: len(v) for k, v in m["event_sets"].items()}}


def describe_variable(variable: str, countries: list[str]) -> dict:
    d = S.series(variable, countries or None)
    years, rows = d["years"], []
    for s in d["series"]:
        pts = [(y, v) for y, v in zip(years, s["values"]) if v is not None]
        if not pts:
            continue
        (y0, v0), (y1, v1) = pts[0], pts[-1]
        rows.append({"country": s["country"], "group": s["group"],
                     "first_year": y0, "first": _r(v0), "last_year": y1,
                     "last": _r(v1), "change": _r(v1 - v0),
                     "mean": _r(sum(v for _, v in pts) / len(pts))})
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r["last"])
    return {"variable": variable, "countries": rows,
            "group_mean_last": {g: _r(sum(v) / len(v)) for g, v in by.items()}}


def rank_countries(variable: str, year: int) -> dict:
    s = S.snapshot(variable, int(year))
    return {"variable": variable, "year": s["year"],
            "ranking": [{"rank": i + 1, "country": r["country"], "group": r["group"],
                         "value": _r(r["value"])} for i, r in enumerate(s["rows"])]}


def local_projection(outcome, shock, horizons, levels, split_by_group) -> dict:
    h = max(1, min(int(horizons), 8))
    r = S.local_projection(outcome, shock, h, 1, levels, split_by_group)
    r["rows"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in x.items()}
                 for x in r["rows"]]
    return r


def event_study(outcome, event_set, custom_events, control, detrend) -> dict:
    r = S.event_study(outcome, event_set, custom_events, control, detrend,
                      3, 5, N_BOOT)
    r["by_event_time"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in x.items()}
                          for x in r["by_event_time"]]
    return r


def convergence_clubs() -> dict:
    c = S.convergence_clubs()
    return {k: c[k] for k in ("variable", "full_t", "converges", "clubs", "divergent")}


def typology() -> dict:
    t = S.typology()
    return {"silhouette": t["silhouette"], "agreement": t["agreement"],
            "clusters": {g: [p["country"] for p in t["points"] if p["cluster"] == g]
                         for g in ("Innovative", "Emerging")}}


def result_section(section_id: str) -> dict:
    return S.result(section_id)


IMPL = {f.__name__: f for f in (list_variables, describe_variable, rank_countries,
                                local_projection, event_study, convergence_clubs,
                                typology, result_section)}


def execute(name: str, args: dict) -> dict:
    if name not in IMPL:
        raise KeyError(f"unknown tool: {name}")
    return IMPL[name](**(args or {}))
