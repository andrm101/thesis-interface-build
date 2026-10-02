<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/wiki/images/cover-methodology.svg" alt="Methodology" width="100%">

The full step-by-step description, with GUI locations and settings, is in the [README → Methodology](https://github.com/andrm101/thesis-interface-build/blob/main/README.md#methodology). This page summarises the design choices.

## Pipeline

1. **Build the level panel.** `build_panel.py` turns the raw World Bank / Eurostat workbook into `data/panel_levels.csv` and applies the [[Data and Audit]] corrections. It also builds a Griliches perpetual-inventory R&D knowledge stock (δ = 15 %).
2. **Screen outliers.** Countries whose mean on a model variable lies beyond |z| > 3 are dropped, which removes Luxembourg (and Malta on the rebuilt panel). Observation-level screening is off by default because its flags fall on 2009 and 2020, which are real shocks.
3. **Explore the data.** Ten families of statistical tests with Benjamini–Hochberg correction, and eleven figure types.
4. **Build the typology.**
   - K-Means with K = 2 on R&D levels.
   - Phillips–Sul convergence clubs: log-t test, club clustering, and Schnurbus et al. merging.
5. **Estimate.**
   - Fixed and random effects with a Hausman test.
   - Cluster-wise comparison with a coefficient-equality test.
   - The lag profile of the R&D effect.
6. **Test time-series properties.**
   - Unit roots: ADF, PP, KPSS, IPS and CIPS (robust to common shocks).
   - Causality: Granger and Dumitrescu–Hurlin.
7. **Handle reverse causality.**
   - Model: Arellano–Bond / Blundell–Bond dynamic panel GMM, with R&D endogenous.
   - Instruments: collapsed and lag-limited.
   - Estimation: two-step with Windmeijer-corrected SEs, plus Hansen and AR tests.
   - Validation: difference GMM matches the Python port of Stata's xtabond2 exactly.
8. **Apply causal designs.**
   - Distance-to-frontier regression.
   - Panel local projections.
   - Callaway–Sant'Anna staggered event study.
   - Synthetic control.
   - Double machine learning, cross-fitted by country.
9. **Build scenarios.** Projections based on the causal estimates.

## Choices worth defending in the viva

| Choice | Why |
|---|---|
| Cluster on R&D **levels**, not growth rates | Growth rates capture volatility, not innovativeness; levels recover the typology exactly |
| System GMM with **collapsed** instruments | N ≈ 23 countries: uncollapsed instruments would outnumber countries and overfit |
| Preferred GMM spec = the one passing **all** diagnostics | Chosen by validity rules stated in advance, not by results |
| Country-held-out cross-validation | Random folds leak a country's own history into the test set |
| Report sensitivity grids (GMM, DML) | Small N makes single estimates fragile; the grids show how fragile |

## Validation

Every estimator is tested by recovering **planted effects** from synthetic data:
- causal designs: event-study ATT, synthetic-control weights, local-projection responses, DML θ and heterogeneity
- dynamic panels: GMM ρ and β under endogeneity
- panel tests: CIPS and Dumitrescu–Hurlin size and power
- convergence: two planted clubs

The suite has 69 tests, including the web API, and runs in CI on Python 3.10 and 3.12. CI also builds the Angular dashboard and smoke-tests its Docker image.

<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/assets/brand-divider.svg" alt="" width="100%">
