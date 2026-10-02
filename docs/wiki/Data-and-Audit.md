<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/wiki/images/cover-data-and-audit.svg" alt="Data and Audit" width="100%">

## Sources

The raw workbook is a World Bank WDI and Eurostat extract covering 27 EU countries, 1998–2023. It contains:
- GDP per capita
- employment and population
- R&D % of GDP and GERD
- researchers
- resident patent applications
- tertiary attainment
- gross savings
- education spending
- productivity per hour

The workbook stays local. The rebuilt panel is committed as `data/panel_levels.csv`.

## What the thesis columns are

| `panel_data.xlsx` column | Verified construction |
|---|---|
| `Y by L` | GDP per capita growth index — **per capita, not per worker** |
| `PIB towards research` | R&D % GDP growth index |
| `Patents per capita` | patents per capita growth index |
| `Labor in research` | researchers / population (level) |
| `Human Capital Proxy` | tertiary attainment growth index — **from misaligned rows** |
| `Savings Percentage` | cannot be reproduced from the workbook |

## Corrections

1. **Human capital rows are shifted.** In the working sheet `AAA`, each country's tertiary series is the *next* country's (Austria ← Belgium, Germany ← Italy, Luxembourg ← Malta, …). This affects 19 of 25 countries, and `Human Capital Proxy` inherited it. Luxembourg's apparent human-capital outlier is Malta's data.
2. **Researchers are ×10** for Czechia and Estonia in the raw sheet (a unit slip). The thesis panel already had these right.
3. **Italy 2011 `Labor` = 100 067.74** in the panel is a transcription typo. The raw employment series is normal.
4. **Savings Percentage** does not match gross savings / GDP. For example, Luxembourg is 0.49 in the panel versus 0.12–0.20 in WDI.

Full detail is in [`docs/DATA_AUDIT.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/DATA_AUDIT.md). The corrections are locked by `tests/test_build_panel.py`.

## Policy variables

| Variable | Coverage 2000–2023 |
|---|---|
| General government balance, % GDP (Eurostat `gov_10dd_edpt1`) | complete |
| OECD implied R&D tax subsidy rate | complete for 22 of 24 countries (Croatia sparse; Greece from 2004) |
| GBARD, million euro (Eurostat NABS 1992 + NABS 2007, spliced) | complete for 16 of 24 (Croatia from 2008, Hungary from 2005, Czechia from 2002) |
| Cyclically adjusted balance, % potential GDP (AMECO `UBLGAP`) | complete for 23 of 24 (Croatia from 2001) |

**GERD cross-check.** Against Eurostat `rd_e_gerdfund`, the workbook's GERD matches within 0.4 % (median), except that the **Slovakia and Slovenia rows are swapped**. This is now corrected in `build_panel.py`.

R&D tax-reform dates are derived from jumps in the subsidy rate, not hand-coded (see [[Key Findings]] §7).

## Policy documents (optional)

The dates of the R&D tax reforms in D2 are derived from the OECD subsidy series. `policy_docs/` checks them against policy documents:
- **Extraction:** Claude extracts each dated reform with a verbatim quote and page.
- **Quote check:** the quote is matched against the document's text (pypdf, fuzzy score ≥ 90).
- **Dates used:** only verified, generosity-raising events become documented dates.
- **Reconciliation:** each country is classed as confirmed, date shift, only documented, or only derived.

The raw documents stay local; the extracted JSON files are committed as the record.

## Maps

The dashboard's maps use Natural Earth 1:50m country outlines (public domain), cut to Europe by `web/scripts/build-europe-map.mjs`. Overseas territories are left out.

## Consequences

- K-Means on levels reproduces the thesis typology exactly.
- Results that used `Human Capital Proxy` should be re-run on the rebuilt panel.
- Luxembourg is still excluded on the rebuilt panel, but for patents per capita (z = 3.3) rather than human capital.

<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/assets/brand-divider.svg" alt="" width="100%">
