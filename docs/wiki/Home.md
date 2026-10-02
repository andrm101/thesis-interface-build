<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/wiki/images/cover-home.svg" alt="Augmented Solow · R&D Heterogeneity Lab" width="100%">

Companion wiki for the thesis *An empirical investigation into the heterogeneous impact of R&D investment on economic productivity across EU member states, 1998–2023*, and for the desktop app, the [[Web Dashboard]] and the reproduction scripts in [andrm101/thesis-interface-build](https://github.com/andrm101/thesis-interface-build).

## Start here

| If you want to… | Read |
|---|---|
| know what the evidence says | [[Key Findings]] |
| understand how the results were produced | [[Methodology]] |
| know which data were used and what was corrected | [[Data and Audit]] |
| regenerate every number, figure and table | [[Reproducing Results]] |
| explore the data and rerun analyses in a browser | [[Web Dashboard]] |
| know what the evidence cannot say yet | [[Limitations and Next Steps]] |

## The argument in brief

The thesis argues that R&D raises productivity in innovation leaders but not yet in catch-up economies. In pooled data, the catch-up economies' low absorptive capacity hides the effect: the *GERD paradox*.

The evidence supports the **direction** of that argument, but the mechanism is different:
- **Productivity drives R&D.** Countries spend more on R&D after they grow richer, which biases naive regressions.
- **Once that feedback is removed, R&D pays off, mainly in the leaders.** This result depends on a narrow set of valid instrument choices.
- **The catch-up economies converge quickly anyway.** Their growth comes mostly from sources other than their own R&D.
- **Policy levers work slowly or only partly.** R&D tax reforms raise R&D spending a little, but not productivity. Public R&D budgets partly crowd out private R&D, above all in the leaders. Fiscal tightening costs output for about two years and leaves no lasting damage. See [[Key Findings]] §7–7c.

![R&D typology](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig1_typology_pca.png)

## Two datasets

| | Thesis panel (Track A) | Rebuilt level panel (Track B) |
|---|---|---|
| File | `panel_data.xlsx` | `data/panel_levels.csv` |
| Content | year-on-year growth indices, as submitted | levels and indices rebuilt from the raw World Bank / Eurostat workbook, with corrections |
| Use | reproduce the thesis as submitted | corrected results, typology, convergence, causal designs |

Everything in this wiki can be regenerated with `python reproduce.py --figures`.

<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/assets/brand-divider.svg" alt="" width="100%">
