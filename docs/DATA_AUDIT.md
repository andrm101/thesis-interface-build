# Data audit — `panel_data.xlsx` vs. the raw workbook

The raw workbook `data/raw/BD_Licenta.xlsx` (World Bank WDI + Eurostat
extracts, 27 EU countries, 1998-2023) was compared cell by cell with the
thesis panel `panel_data.xlsx`. `build_panel.py` rebuilds the panel from the
raw sheets and writes `data/panel_levels.csv`; `tests/test_build_panel.py`
locks the findings below in place.

## How the thesis columns were built

| `panel_data.xlsx` column | Construction (verified) | Status |
|---|---|---|
| `Y by L` | GDP per capita (const. 2015 US$) growth index, prev. yr = 100 — **per capita, not per worker** | ✔ exact |
| `PIB towards research` | R&D expenditure % GDP, growth index | ✔ exact |
| `Patents per capita` | resident patent applications per capita, growth index | ✔ exact |
| `Labor in research` | researchers / population (level) | ✔ exact for 21/23; see (2) |
| `Human Capital Proxy` | tertiary attainment growth index — **from misaligned rows**, see (1) | ✘ wrong for 19/23 countries |
| `Labor` | employment growth index | ✔ 17/23; Italy 2011 typo, see (3) |
| `Savings Percentage` | not reproducible from the workbook, see (4) | ? |

## Findings

1. **Human capital rows are shifted.** In sheet `AAA`, block "Human C", each
   country row holds the *next* row's series from sheet `Tertiary Educ`
   (Austria ← Belgium, Belgium ← Bulgaria, Germany ← Italy, Luxembourg ←
   Malta, …; 19 of 25 countries). `Human Capital Proxy` equals the growth
   index of this block exactly (corr = 1.000) and correlates only 0.17 with
   the correctly aligned series. Luxembourg's extreme human-capital z-score
   (+3.3) was in fact Malta's data.
   *Fix:* read `Tertiary Educ` directly.

2. **Researchers ×10 for Czechia and Estonia** in sheet `Researchers`
   (unit slip). The thesis panel already had the corrected values.
   *Fix:* divide by 10 in `build_panel.py`.

3. **Italy 2011 `Labor` = 100 067.74** in the panel; raw employment moves
   normally (36.75 → 36.84 m), so this is a transcription error in the
   panel (should be ≈ 100.24).

4. **`Savings Percentage` cannot be reproduced.** Gross savings / GDP (both
   current US$, WDI) correlates 0.59 with it; Luxembourg is 0.49 in the
   panel vs. 0.12-0.20 from WDI. The rebuilt panel uses gross savings /
   GDP and flags the difference.

5. **Most thesis variables are growth indices, not levels.** Level
   variables are now available (`RD_pct_GDP`, `Researchers_per_1000_emp`,
   `Patents_per_million`, `Tertiary_share`, `Y_per_worker`,
   `Frontier_gap`, …).

## Consequences

| Analysis | On `panel_data.xlsx` | On rebuilt levels |
|---|---|---|
| K-Means, K = 2, R&D variables | Mixed clusters, silhouette 0.24 | **Exactly the thesis typology** (Innovative = AT, BE, DK, FI, FR, DE, NL, SE), silhouette 0.37 |
| Z-score screen (|z| > 3) | Luxembourg via human capital (Malta's data) | Luxembourg via patents per capita (z = 3.3); Malta via tertiary-growth volatility |
| Phillips-Sul clubs | only on chained growth paths | on log output per worker levels (see Tab 4) |

Results that used `Human Capital Proxy` (panel FE, interaction and
threshold models, clustering) should be re-run on the rebuilt panel.
