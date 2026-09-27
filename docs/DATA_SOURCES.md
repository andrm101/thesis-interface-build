# Candidate data sources

Open, citable sources that would extend `panel_data.xlsx`. The shipped panel
is almost entirely **year-on-year growth indices (previous year = 100)** —
`Y by L`, `PIB towards research`, `Human Capital Proxy`, `Patents*`,
`Population`, `Labor` all hover around 100 and dip in 2009/2020 — with only
`Labor in research` and `Savings Percentage` in levels. Level series are
needed for (a) a meaningful typology, (b) distance-to-frontier measures, and
(c) any "policy change" treatment.

Codes marked ✔ were verified against the publisher in September 2026; the
rest are the publisher's standard identifiers and should be checked when
downloading.

## 1 · R&D and innovation inputs

| Variable | Source | Code / table | Why |
|---|---|---|---|
| GERD, % of GDP, by sector (business / government / higher-ed) | Eurostat | `rd_e_gerdtot` | Proper R&D **intensity level**; BERD vs GOVERD split tests whether private or public R&D drives productivity |
| Government budget allocations for R&D (GBARD), % GDP & by objective | Eurostat ✔ | `gba_nabsfin07` | Direct **fiscal R&D policy** lever; the treatment variable for policy-change designs |
| Implied R&D tax-subsidy rate (1 − B-index), large/SME, profit/loss | OECD R&D Tax Incentives ✔ | `DSD_RDTAX@DF_RDSUB` (OECD Data Explorer), B-index 2000-2025 | Timing and generosity of **R&D tax-credit reforms** → event-study treatment dates |
| Business R&D funded by government (direct vs. indirect support) | OECD | Main Science & Technology Indicators (MSTI) | Separates grants from tax relief |
| R&D personnel & researchers, FTE, % of labour force | Eurostat | `rd_p_persocc` | Level replacement for `Labor in research` |
| EPO patent applications per million inhabitants | Eurostat | `pat_ep_ntot` | Level replacement for `Patents per capita` |
| European Innovation Scoreboard summary index | European Commission (EIS) | annual scoreboard files | Validates the Innovative/Emerging typology externally |

## 2 · Productivity, capital and human capital

| Variable | Source | Code / table | Why |
|---|---|---|---|
| Real GDP, capital stock, TFP level, human-capital index (1950-2023) | Penn World Table 11.0 ✔ | `rgdpo`, `cn`, `ctfp`, `rtfpna`, `hc` | Internationally comparable **levels**; TFP gap to the frontier (US or EU leader) for Aghion-Howitt tests |
| Labour productivity per hour / per person | Eurostat | `nama_10_lp_ulc` | Productivity level in PPS |
| Tertiary attainment, 25-64 | Eurostat | `edat_lfse_03` | Absorptive-capacity moderator |
| Growth accounting by industry (capital, ICT, intangibles) | EUKLEMS & INTANProd (LLEE) | country-industry files | Intangible investment as an alternative innovation input |

## 3 · Fiscal policy

| Variable | Source | Code / table | Why |
|---|---|---|---|
| Government revenue, expenditure, balance, debt | Eurostat | `gov_10a_main` | Fiscal stance controls; consolidation episodes |
| Expenditure by function (COFOG: education, R&D, economic affairs) | Eurostat | `gov_10a_exp` | Education & R&D spending shares |
| Cyclically-adjusted / structural balance | European Commission AMECO | AMECO database | Identifies **discretionary** fiscal changes (vs. automatic stabilisers) |
| Narrative fiscal consolidation episodes | IMF (Devries et al. 2011; updated by Alesina et al.) | public dataset | Exogenous fiscal-shock dates for local projections |

## 4 · EU funds and institutions

| Variable | Source | Code / table | Why |
|---|---|---|---|
| Cohesion Policy payments by country / year / theme | EU Cohesion Open Data Platform | cohesiondata.ec.europa.eu | Catch-up funding shock; R&D-themed ERDF |
| Horizon 2020 / Horizon Europe grants | CORDIS open data | project & participant files | EU-level R&D funding per country |
| Government effectiveness, regulatory quality, rule of law | World Bank WGI | `GE.EST`, `RQ.EST`, `RL.EST` | Institutional absorptive capacity |
| R&D expenditure % GDP (cross-check) | World Bank WDI | `GB.XPD.RSDV.GD.ZS` | Fallback / cross-validation for GERD |

## Suggested order of work

1. GERD % GDP + GBARD + R&D tax-subsidy rate (the policy variables).
2. PWT 11.0 TFP level and `hc` (frontier gap, absorptive capacity).
3. `gov_10a_main` + AMECO structural balance (fiscal stance).
4. Cohesion / Horizon funding (catch-up shocks).

Keep the current growth-index columns — they are the right form for
local projections and unit-root-safe regressions — but add the levels next
to them with a clear suffix (e.g. `GERD_pct_gdp`).

### Data-quality note

`Labor` for **Italy, 2011** is `100067.74` — almost certainly a misplaced
decimal (`100.06774`). The Tab 1 z-score screen flags it when *Vars = All
numeric* and *Level = Observations/Both*.

## Sources

- Eurostat, [Government budget allocations for R&D (GBARD) — Statistics Explained](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Government_budget_allocations_for_R&oldid=573250)
- OECD, [Implied tax subsidy rates on R&D expenditures — Data Explorer](https://data-explorer.oecd.org/vis?df%5Bds%5D=DisseminateFinalDMZ&df%5Bid%5D=DSD_RDTAX%40DF_RDSUB&df%5Bag%5D=OECD.STI.STP)
- OECD, [Tax incentives for R&D — Corporate Tax Statistics 2026](https://www.oecd.org/en/publications/corporate-tax-statistics-2026_73af6222-en/full-report/tax-incentives-for-research-and-development_01cf85d0.html)
- Groningen Growth and Development Centre, [Penn World Table 11.0](https://www.rug.nl/ggdc/productivity/pwt/?lang=en)
