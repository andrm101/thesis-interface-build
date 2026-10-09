"""
export_data.py — every number the explainer video shows, computed by the
same functions as reproduce.py and the dashboard (nothing typed by hand).

    python assets/motion/export_data.py      → assets/motion/data.json
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import panel_tests
from api import service as S

W, H = 760, 640                          # map box in the video (px)
LON0, LAT0 = math.radians(10), math.radians(52)


def laea(lon, lat):
    l, p = math.radians(lon) - LON0, math.radians(lat)
    k = math.sqrt(2 / (1 + math.sin(LAT0) * math.sin(p)
                       + math.cos(LAT0) * math.cos(p) * math.cos(l)))
    return (k * math.cos(p) * math.sin(l),
            -k * (math.cos(LAT0) * math.sin(p)
                  - math.sin(LAT0) * math.cos(p) * math.cos(l)))


def europe_paths(panel: set[str]) -> list[dict]:
    with open(os.path.join(ROOT, "web", "public", "europe.json"), encoding="utf-8") as fh:
        geo = json.load(fh)
    feats = []
    for f in geo["features"]:
        rings = [[laea(*pt) for pt in ring] for poly in f["geometry"]["coordinates"]
                 for ring in poly]
        feats.append((f["properties"]["name"], rings))
    # frame: fit the panel countries (plus a margin), not Iceland or Turkey
    xs = [x for n, rs in feats if n in panel for r in rs for x, _ in r]
    ys = [y for n, rs in feats if n in panel for r in rs for _, y in r]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    s = min(W / (x1 - x0), H / (y1 - y0)) * 0.94
    ox, oy = (W - s * (x1 - x0)) / 2, (H - s * (y1 - y0)) / 2
    out = []
    for name, rings in feats:
        d = ""
        for r in rings:
            pts = [(ox + s * (x - x0), oy + s * (y - y0)) for x, y in r]
            if all(px < -200 or px > W + 200 or py < -200 or py > H + 200 for px, py in pts):
                continue
            d += "M" + "L".join(f"{px:.1f},{py:.1f}" for px, py in pts) + "Z"
        if d:
            out.append({"name": name, "d": d})
    return out


def r(x, n=3):
    return None if x is None else round(float(x), n)


def main():
    df = S.df()
    countries = sorted(df["Country"].unique())
    typ = S.typology()
    clubs = S.convergence_clubs()
    series = S.series("RD_pct_GDP")
    cluster = {p["country"]: p["cluster"] for p in typ["points"]}   # countries with data
    groups = S.groups()            # thesis typology; K-Means reproduces it (agreement 1.0)
    club = {p["country"]: p["club"] for p in clubs["paths"]}
    rd = {s["country"]: [r(v, 3) for v in s["values"]] for s in series["series"]}

    gr_rd_y = panel_tests.dumitrescu_hurlin(df, "RD_pct_GDP", "log_Y_per_worker", 1, difference=True)
    gr_y_rd = panel_tests.dumitrescu_hurlin(df, "log_Y_per_worker", "RD_pct_GDP", 1, difference=True)

    es = S.event_study("RD_pct_GDP", "R&D tax reforms", "", "Not yet treated", False, 3, 5, 199)
    gb = S.local_projection("RD_pct_GDP", "GBARD_pct_GDP", 5, 1, True, False)["rows"]
    ng = S.local_projection("NonGBARD_RD_pct_GDP", "GBARD_pct_GDP", 5, 1, True, False)["rows"]
    cab = S.local_projection("Y_per_worker", "CAB_pct_potGDP", 5, 1, False, False)["rows"]

    data = {
        "years": series["years"],
        "countries": [{"name": c, "group": groups[c], "cluster": cluster.get(c),
                       "club": club.get(c)}
                      for c in countries],
        "rd": rd,
        "clubs": {"years": clubs["years"], "n": len(clubs["clubs"]),
                  "paths": [{"country": p["country"], "club": p["club"],
                             "values": [r(v, 4) for v in p["values"]]} for p in clubs["paths"]]},
        "typology": {"silhouette": r(typ["silhouette"], 2), "agreement": r(typ["agreement"], 2)},
        "granger": {"rd_to_y_p": r(gr_rd_y.p_Z_tilde, 3), "y_to_rd_p": r(gr_y_rd.p_Z_tilde, 3)},
        "tax_reforms": {"n": len(es["treated"]), "att": r(es["overall_post"], 1),
                        "se": r(es["overall_post_se"], 1), "pretrend_p": r(es["pretrend_p"], 2)},
        "gbard": {"multiplier_h0": r(gb[0]["beta"], 2), "private_h0": r(ng[0]["beta"], 2)},
        "fiscal": {"y_h0": r(cab[0]["beta"], 2), "y_h1": r(cab[1]["beta"], 2),
                   "y_h3": r(cab[3]["beta"], 2)},
        "map": {"w": W, "h": H, "paths": europe_paths(set(countries))},
    }
    out = os.path.join(HERE, "data.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out) // 1024} kB)")
    print({k: data[k] for k in ("typology", "granger", "tax_reforms", "gbard", "fiscal")})


if __name__ == "__main__":
    main()
