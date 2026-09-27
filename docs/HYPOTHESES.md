# Hypothesis catalogue

Testable hypotheses drawn from the growth / innovation literature, each
mapped to where the app tests it **now** (✅), what it could test with the
**current data but new code** (🔧), and what needs **new data** from
[`DATA_SOURCES.md`](DATA_SOURCES.md) (📥). Guiding principle: many
complementary tests, with multiple-testing corrections (Benjamini-Hochberg)
wherever a family is run over many variables.

| # | Hypothesis | Literature | How to test | Status |
|---|---|---|---|---|
| H1 | Savings (physical-capital accumulation) raises productivity | Solow (1956); Mankiw, Romer & Weil (1992) | Panel FE on `Y by L` | ✅ Tab 3 |
| H2 | Human capital raises productivity (augmented Solow) | Mankiw, Romer & Weil (1992) | Panel FE; Tab 2 correlations | ✅ Tab 3, Tab 2 |
| H3 | Aggregate R&D–productivity link is negative/weak but positive among leaders (GERD paradox) | Thesis; Griffith, Redding & Van Reenen (2004) | Cluster-wise FE; pooled vs. within correlations (Simpson check); system GMM with R&D endogenous | ✅ Tab 3/8, Tab 2, Tab 14 Dynamic GMM |
| H4 | R&D returns depend on absorptive capacity (human capital) | Cohen & Levinthal (1990); Griffith et al. (2004) | R&D × human-capital interaction; threshold regression | ✅ Tab 12 |
| H5 | Conditional β- and σ-convergence | Barro & Sala-i-Martin (1992) | Cross-sectional OLS on initial level; dispersion over time | ✅ Tab 4 |
| H6 | Countries converge in **clubs**, not globally | Phillips & Sul (2007) | log-t test + clustering algorithm | ✅ Tab 4 (growth-path clubs); 📥 PWT levels for level clubs |
| H7 | R&D pays off more near the technological frontier; imitation pays off far from it | Acemoglu, Aghion & Zilibotti (2006); Aghion & Howitt (2006) | R&D × distance-to-frontier interaction; DML CATE in gap | ✅ Tab 14 (gap from rebuilt level panel) |
| H8 | R&D leads productivity with a lag; productivity does not lead R&D | Griliches (1979) knowledge-stock lag | Lead-lag correlogram; panel Granger (Dumitrescu-Hurlin) | ✅ Tab 2 lead-lag, Tab 11 Granger & Dumitrescu-Hurlin |
| H9 | Foreign R&D spills over through trade | Coe & Helpman (1995) | Trade-weighted foreign R&D stock in FE | 📥 trade weights |
| H10 | Productivity series share common shocks (cross-sectional dependence) | Pesaran (2004, 2007) | CD test; CIPS unit root; Driscoll-Kraay SE | ✅ Tab 2 CD, Tab 10 CIPS, Tab 12 DK |
| H11 | Productivity growth broke after 2008-09 and 2020 | Productivity-slowdown literature | Chow tests at candidate years; year effects | ✅ Tab 2 breaks |
| H12 | R&D tax-credit introductions raise business R&D and, with a lag, productivity | Bloom, Griffith & Van Reenen (2002); Dechezleprêtre et al. (2023) | Staggered event study (Callaway-Sant'Anna), synthetic control | ✅ Tab 14 design (custom events); 📥 OECD B-index dates |
| H13 | Public R&D (GBARD) crowds in private R&D | David, Hall & Toole (2000) | FE / local projections of BERD on GBARD | 🔧 Tab 14 LPs ready; 📥 GBARD, BERD |
| H14 | Fiscal consolidations reduce R&D and long-run productivity (hysteresis) | Fatás & Summers (2018); Alesina et al. (2019) | Local projections on narrative consolidation shocks | 🔧 Tab 14 LPs ready; 📥 IMF episodes, AMECO |
| H15 | EU Cohesion funds accelerate catch-up | Becker, Egger & von Ehrlich (2010) | Local projections / synthetic control on fund inflows | ✅ Tab 14 designs; 📥 Cohesion data |
| H16 | Institutional quality conditions R&D returns | Rodrik, Subramanian & Trebbi (2004) | R&D × WGI interaction | 📥 WGI |
| H17 | Country effects dominate variation (entity FE justified) | — | ANOVA / Kruskal-Wallis, η², between/within decomposition | ✅ Tab 2 |
| H18 | Innovative and Emerging groups differ in distribution, not just mean | — | Welch, Mann-Whitney, KS, Levene, Cohen's d (FDR) | ✅ Tab 2 |

## References (short form)

Acemoglu, Aghion & Zilibotti (2006) *JEEA*; Aghion & Howitt (2006) *JEEA*;
Alesina, Favero & Giavazzi (2019) *Austerity*; Barro & Sala-i-Martin (1992)
*JPE*; Becker, Egger & von Ehrlich (2010) *JPubE*; Bloom, Griffith & Van
Reenen (2002) *JPubE*; Coe & Helpman (1995) *EER*; Cohen & Levinthal (1990)
*ASQ*; David, Hall & Toole (2000) *Research Policy*; Callaway & Sant'Anna (2021) *J. Econometrics*;
Chernozhukov et al. (2018) *Econometrics J.*; Jordà (2005) *AER*; Abadie,
Diamond & Hainmueller (2010) *JASA*; Dumitrescu & Hurlin (2012) *Econ.
Modelling*; Ferman & Pinto (2021) *Quant. Econ.*;
Dechezleprêtre, Einiö,
Martin, Nguyen & Van Reenen (2023) *AEJ: Policy*; Fatás & Summers (2018)
*JIE*; Griffith, Redding & Van Reenen (2004) *REStat*; Griliches (1979)
*Bell J. Econ.*; Mankiw, Romer & Weil (1992) *QJE*; Pesaran (2004) CESifo WP
1229, (2007) *J. Appl. Econometrics*; Phillips & Sul (2007) *Econometrica*,
(2009) *J. Appl. Econometrics*; Schnurbus, Haupt & Meier (2017) *Oxford
Bull. Econ. Stat.*;
Rodrik, Subramanian & Trebbi (2004) *J. Econ. Growth*; Solow (1956) *QJE*.
