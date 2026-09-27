"""
build_panel.py — Rebuild the thesis panel in LEVELS from the raw workbook.

    python build_panel.py                      # data/raw/BD_Licenta.xlsx
    python build_panel.py path/to/raw.xlsx -o data/panel_levels.csv

The raw workbook (World Bank WDI + Eurostat extracts) holds each series as
a wide country × year block whose header row contains "TIME" and whose
rows carry ISO-3 country codes. This script reads the top (raw) block of
each sheet — and the labelled model-input blocks of sheet "AAA" — and
writes a long Country × Year panel with level variables plus derived
intensities, log levels and year-on-year growth indices (prev. year = 100,
the form used by panel_data.xlsx).

Data corrections (see docs/DATA_AUDIT.md):
  • Tertiary attainment is read from sheet "Tertiary Educ", NOT the
    "Human C" block of sheet "AAA": that block's values are shifted one
    row against its country codes (Austria holds Belgium's series, …),
    and panel_data.xlsx's `Human Capital Proxy` inherited the shift.
  • Researchers for Czechia and Estonia are 10× too large in sheet
    "Researchers" (unit slip); divided by 10, matching panel_data.xlsx.

The output also carries thesis-compatible columns (`Y by L`,
`PIB towards research`, `Patents per capita`, `Human Capital Proxy`,
`Labor in research`, `Savings Percentage`, `Labor`, `Population`) so every
tab of the app runs on it unchanged.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import openpyxl
import pandas as pd

ISO = re.compile(r"^[A-Z]{3}$")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(HERE, "data", "raw", "BD_Licenta.xlsx")
DEFAULT_OUT = os.path.join(HERE, "data", "panel_levels.csv")

# (sheet, block label or None for the first block) → output column
SERIES = {
    "GDP_pc_2015usd":      ("GDP per Capita + Formule", None),
    "GDP_usd":             ("GDP", None),
    "Population":          ("Pop", None),
    "Employment":          ("AAA", "Labor"),
    "LF_participation_pct": ("Labor", None),
    "RD_pct_GDP":          ("AAA", "K research"),
    "GERD_mEUR":           ("GERD", None),
    "Researchers":         ("AAA", "L Research"),
    "Patent_apps":         ("AAA", "Pa"),
    "Tertiary_pct":        ("Tertiary Educ", None),
    "Gross_savings_usd":   ("Savings", None),
    "Educ_spend_pct_GDP":  ("Gov Spending Educ", None),
    "GDP_growth_pct":      ("GDP Growth", None),
    "Prod_per_hour_growth_pct": ("Productivity", None),
}


def _num(v):
    if v is None or v == "..":
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _year(v):
    m = re.match(r"^\s*(\d{4})", str(v)) if v is not None else None
    return int(m.group(1)) if m else None


def read_block(rows: list[tuple], label: str | None) -> pd.DataFrame:
    """Return a Country-code × Year frame for the first block (or the block
    whose header row starts with *label*)."""
    start = None
    for i, r in enumerate(rows):
        cells = [str(c).strip() if c is not None else "" for c in r[:3]]
        if "TIME" in cells and (label is None or cells[0] == label):
            start = i
            break
    if start is None:
        raise KeyError(f"block {label!r} not found")
    hdr = rows[start]
    ycols = {j: _year(v) for j, v in enumerate(hdr) if _year(v)}
    # stop at the next TIME header block to the right, if any
    first_y = min(ycols)
    ycols = {j: y for j, y in ycols.items()
             if j < first_y + 40 and list(ycols.values()).count(y) >= 1}
    seen, keep = set(), {}
    for j, y in sorted(ycols.items()):
        if y in seen:
            break
        seen.add(y); keep[j] = y
    recs = {}
    for r in rows[start + 1:]:
        a = str(r[0]).strip() if r[0] is not None else ""
        b = str(r[1]).strip() if len(r) > 1 and r[1] is not None else ""
        code = b if ISO.match(b) else a if ISO.match(a) else None
        if code is None:
            if recs:
                break       # end of block
            continue
        if code in recs:
            continue
        recs[code] = {y: _num(r[j]) for j, y in keep.items()}
    return pd.DataFrame(recs).T.sort_index()


def country_names(rows) -> dict[str, str]:
    """ISO → name from the WDI long sheet ('Data')."""
    out = {}
    for r in rows[1:]:
        if r[1] and ISO.match(str(r[1])) and r[0]:
            out[str(r[1])] = str(r[0]).strip()
    return out


def build(raw_path: str = DEFAULT_RAW) -> pd.DataFrame:
    wb = openpyxl.load_workbook(raw_path, read_only=True, data_only=True)
    cache = {}

    def rows(sheet):
        if sheet not in cache:
            cache[sheet] = list(wb[sheet].iter_rows(values_only=True))
        return cache[sheet]

    names = country_names(rows("Data"))
    # Wrong-by-10 researcher counts (unit slip in the raw sheet).
    researcher_fix = {"CZE": 0.1, "EST": 0.1}
    names.update({"SVK": "Slovakia"})
    frames = []
    for col, (sheet, label) in SERIES.items():
        blk = read_block(rows(sheet), label)
        if col == "Researchers":
            for code, f in researcher_fix.items():
                if code in blk.index:
                    blk.loc[code] *= f
        s = blk.stack(future_stack=True).rename(col)
        s.index.names = ["Code", "Year"]
        frames.append(s)
    df = pd.concat(frames, axis=1).reset_index()
    df = df[df["Code"].isin(names) & (df["Code"] != "EUU")]
    df.insert(0, "Country", df["Code"].map(names))
    df["Year"] = df["Year"].astype(int)
    return add_derived(df.sort_values(["Country", "Year"])
                       .reset_index(drop=True))


RD_DEPRECIATION = 0.15   # Griliches / OECD convention for R&D capital


def knowledge_stock(rd: pd.Series, delta: float = RD_DEPRECIATION,
                    g_years: int = 5) -> pd.Series:
    """Perpetual-inventory R&D capital for one country's real R&D flow:
    K_t = (1 − δ)·K_{t−1} + R_t,  K_0 = R_0 / (g + δ),
    g = average growth of R over the first *g_years* (clipped to [0, 0.2]
    so a falling or erratic start cannot give a negative or huge K_0)."""
    r = rd.to_numpy(float)
    out = np.full_like(r, np.nan)
    ok = np.flatnonzero(np.isfinite(r) & (r > 0))
    if len(ok) == 0:
        return pd.Series(out, index=rd.index)
    i0 = ok[0]
    first = r[i0:i0 + g_years + 1]
    first = first[np.isfinite(first)]
    g = (np.clip((first[-1] / first[0]) ** (1 / (len(first) - 1)) - 1, 0, 0.2)
         if len(first) > 1 else 0.05)
    k = r[i0] / (g + delta)
    out[i0] = k
    for t in range(i0 + 1, len(r)):
        k = (1 - delta) * k + (r[t] if np.isfinite(r[t]) else 0.0)
        out[t] = k
    return pd.Series(out, index=rd.index)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Tertiary_share"] = d["Tertiary_pct"] / 100
    # Real R&D expenditure and the Griliches knowledge stock
    d["RD_exp_2015usd"] = (d["RD_pct_GDP"] / 100 * d["GDP_pc_2015usd"]
                           * d["Population"])
    d["RD_stock"] = (d.sort_values("Year").groupby("Country")["RD_exp_2015usd"]
                     .transform(knowledge_stock))
    d["GDP_2015usd"] = d["GDP_pc_2015usd"] * d["Population"]
    d["Y_per_worker"] = d["GDP_2015usd"] / d["Employment"]
    d["log_Y_per_worker"] = np.log(d["Y_per_worker"])
    d["log_GDP_pc"] = np.log(d["GDP_pc_2015usd"])
    d["Savings_rate"] = d["Gross_savings_usd"] / d["GDP_usd"]
    d["Researchers_per_1000_emp"] = d["Researchers"] / d["Employment"] * 1e3
    d["RD_stock_per_worker"] = d["RD_stock"] / d["Employment"]
    d["log_RD_stock_per_worker"] = np.log(d["RD_stock_per_worker"])
    d["Patents_per_million"] = d["Patent_apps"] / d["Population"] * 1e6
    # Distance to the productivity frontier: mean log Y/L of the top three
    # countries each year (a single-country max would be Luxembourg).
    front = d.groupby("Year")["log_Y_per_worker"].transform(
        lambda s: s.nlargest(3).mean())
    d["Frontier_gap"] = front - d["log_Y_per_worker"]
    # Year-on-year growth indices, previous year = 100
    g = d.groupby("Country")
    for c in ("Y_per_worker", "GDP_pc_2015usd", "RD_pct_GDP",
              "Researchers", "Patents_per_million", "Tertiary_share"):
        d[f"{c}_idx"] = g[c].pct_change(fill_method=None) * 100 + 100

    # Thesis-compatible columns (same construction as panel_data.xlsx,
    # from corrected inputs) so the app's default variable picks work.
    idx = lambda c: g[c].pct_change(fill_method=None) * 100 + 100  # noqa: E731
    d["Y by L"] = d["GDP_pc_2015usd_idx"]          # GDP per capita, as thesis
    d["PIB towards research"] = d["RD_pct_GDP_idx"]
    d["Patents per capita"] = d["Patents_per_million_idx"]
    d["Human Capital Proxy"] = d["Tertiary_share_idx"]
    d["Labor in research"] = d["Researchers"] / d["Population"]
    d["Savings Percentage"] = d["Savings_rate"]
    d["Labor"] = idx("Employment")
    d["Population_idx"] = idx("Population")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="?", default=DEFAULT_RAW)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    df = build(a.raw)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, {df['Country'].nunique()} "
          f"countries, {df['Year'].min()}-{df['Year'].max()}, "
          f"{df.shape[1]} columns")


if __name__ == "__main__":
    main()
