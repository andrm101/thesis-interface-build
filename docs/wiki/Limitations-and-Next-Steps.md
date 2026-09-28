# Limitations and Next Steps

## Limitations

- **Small N.** 24 countries over 24 years. GMM, clustering and group comparisons all have limited power, so the sensitivity grids matter more than any single estimate.
- **Instrument sensitivity.** The positive GMM effect of R&D holds only in the shallow-lag specification that passes every diagnostic.
- **The thesis panel stores growth indices.** Level results come from the rebuilt panel. Figures based on `Human Capital Proxy` from the submitted panel are affected by the row shift.
- **Unit roots.** Log productivity is I(1) (CIPS), so level regressions rely on the dynamic specifications.
- **Policy data are partial.**
  - R&D tax incentives and the headline fiscal balance are complete for 2000–2023.
  - Government R&D budgets (GBARD) lack **2008–2016**, because the NABS 2007 export was Eurostat's ten-year default view.
  - The cyclically adjusted balance is not yet included.
  - Coverage per country is in `data/policy_coverage.csv`.

## Next steps

1. **Complete the policy data.**
   - Re-export `gba_nabsfin07` with *all* years (Customize dataset → TIME → select all). This fills GBARD 2008–2016 and enables the public-R&D crowding-in tests.
   - Add AMECO's cyclically adjusted balance (`UBLGAP`) for a causal fiscal test.
   - Add Eurostat `rd_e_gerdfund` / `rd_e_fundgerd` (R&D by source of funds). See [`docs/DATA_SOURCES.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/DATA_SOURCES.md).
2. **External level data.** Penn World Table TFP levels would give a frontier gap independent of this panel.
3. **Hypotheses.** Of the eighteen literature-based hypotheses in [`docs/HYPOTHESES.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/HYPOTHESES.md):
   - fourteen are testable now (H15, Cohesion funds, only as a ready design);
   - H14 (fiscal consolidation) is partial, pending the cyclically adjusted balance;
   - H9, H13 and H16 need further data: trade weights, the full GBARD series with R&D by source of funds, and governance indicators.
