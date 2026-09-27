# Limitations and Next Steps

## Limitations

- **Small N.** 24 countries over 24 years. GMM, clustering and group comparisons all have limited power, so the sensitivity grids matter more than any single estimate.
- **Instrument sensitivity.** The positive GMM effect of R&D holds only in the shallow-lag specification that passes every diagnostic.
- **The thesis panel stores growth indices.** Level results come from the rebuilt panel. Figures based on `Human Capital Proxy` from the submitted panel are affected by the row shift.
- **Unit roots.** Log productivity is I(1) (CIPS), so level regressions rely on the dynamic specifications.
- **No policy variables yet.** The event study and local projections are ready, but there is no data yet on R&D budgets or R&D tax-incentive reforms.

## Next steps

1. **Policy data.**
   - Eurostat government budget allocations for R&D (`gba_nabsfin07`).
   - OECD R&D tax-incentive rates and introduction dates (`DSD_RDTAX@DF_RDSUB`).
   - AMECO structural balances.

   With these, Tab 14's event study and local projections can test real R&D and fiscal policy changes, for example `Poland:2016, Slovakia:2015`. See [`docs/DATA_SOURCES.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/DATA_SOURCES.md).
2. **External level data.** Penn World Table TFP levels would give a frontier gap independent of this panel.
3. **Hypotheses.** Fifteen of the eighteen literature-based hypotheses in [`docs/HYPOTHESES.md`](https://github.com/andrm101/thesis-interface-build/blob/main/docs/HYPOTHESES.md) are testable now. The remaining three need the data above.
