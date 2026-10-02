<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/wiki/images/cover-limitations.svg" alt="Limitations and Next Steps" width="100%">

## Limitations

- **Small N.** 24 countries over 24 years. GMM, clustering and group comparisons all have limited power, so the sensitivity grids matter more than any single estimate.
- **Instrument sensitivity.** The positive GMM effect of R&D holds only in the shallow-lag specification that passes every diagnostic.
- **The thesis panel stores growth indices.** Level results come from the rebuilt panel. Figures based on `Human Capital Proxy` from the submitted panel are affected by the row shift.
- **Unit roots.** Log productivity is I(1) (CIPS), so level regressions rely on the dynamic specifications.
- **Policy data.** Tax incentives, the headline and cyclically adjusted balances, and GBARD are near-complete for 2000–2023 (coverage per country in `data/policy_coverage.csv`). R&D by source of funds is proxied as total R&D minus GBARD.

## Next steps

1. **R&D by source of funds.** Export Eurostat `rd_e_gerdfund` with *Source of funds* = business enterprise and government, so the crowding-in test (D5) can use measured business-funded R&D instead of the proxy. See [`docs/DATA_SOURCES.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/DATA_SOURCES.md).
2. **External level data.** Penn World Table TFP levels would give a frontier gap independent of this panel.
3. **Hypotheses.** Of the eighteen literature-based hypotheses in [`docs/HYPOTHESES.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/HYPOTHESES.md):
   - sixteen are tested (H15, Cohesion funds, only as a ready design);
   - H9 and H16 need further data: trade weights and governance indicators.
4. **Dashboard.** Add the GMM specification grid and synthetic control to the [[Web Dashboard]]; both are in the desktop app only for now.
5. **Documented reform dates.** Run the policy-document extraction on the OECD country profiles. That gives D2b, an event study on dates confirmed by the documents, and a country-by-country check of the derived dates (`data/policy_events/reconciliation.csv`).

<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/assets/brand-divider.svg" alt="" width="100%">
