# Reproducing Results

```bash
pip install -r requirements.txt
python reproduce.py --figures      # ≈ 2 minutes
```

This writes three outputs:
- [`results/RESULTS.md`](https://github.com/andrm101/thesis-interface-build/blob/main/results/RESULTS.md): every headline number, sections A1–A6, B1–B13, C1 and D1–D4.
- [`results/figures/`](https://github.com/andrm101/thesis-interface-build/blob/main/results/figures): eight figures, PNG and PDF.
- [`results/tables/`](https://github.com/andrm101/thesis-interface-build/blob/main/results/tables): LaTeX `booktabs` tables for `\input{}`.

All random steps are seeded, so the numbers are identical from run to run.

## Figures

| File | Shows |
|---|---|
| `fig1_typology_pca` | K-Means typology on R&D levels |
| `fig2_convergence_clubs` | Phillips–Sul clubs |
| `fig3_jcurve_lags` | R&D effect by lag and group |
| `fig4_event_study` | EU-accession event study, four variants |
| `fig5_gmm_specifications` | GMM R&D coefficient across the specification grid |
| `fig6_dml_sensitivity` | DML frontier-gap slope by sample and learner |
| `fig7_beta_convergence` | β-convergence by typology |
| `fig8_tax_reforms` | R&D tax-incentive reforms, event study |

The figures use a colour-blind-validated palette, and each series has its own marker shape, so they print legibly in greyscale.

## In the app

```bash
python main.py                         # thesis panel
python main.py data/panel_levels.csv   # rebuilt level panel
```

A Windows build (`EU-Innovation-Panel-windows.zip`) is attached to each [GitHub release](https://github.com/andrm101/thesis-interface-build/releases). The [README](https://github.com/andrm101/thesis-interface-build/blob/main/README.md#reaching-every-result) maps every result to its tab and button.
