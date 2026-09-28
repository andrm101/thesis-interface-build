# Key Findings

Section IDs (A1…C1) refer to [`results/RESULTS.md`](https://github.com/andrm101/thesis-interface-build/blob/main/results/RESULTS.md), which `python reproduce.py` regenerates. Significance: \* 10 %, \*\* 5 %, \*\*\* 1 %.

## 1. Two R&D regimes are real

Clustering countries on R&D **levels** recovers the thesis typology exactly (B2, silhouette 0.36). The levels used are R&D % GDP, researchers per 1,000 employed, patents per million and tertiary share.
- **Innovative:** Austria, Belgium, Denmark, Finland, France, Germany, Netherlands, Sweden.
- **Emerging:** the 16 Eastern and Southern economies.

The original clustering used the thesis panel's growth rates, which measure how volatile R&D growth is rather than how innovative a country is. It matched only 77 % (A4).

![Typology](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig1_typology_pca.png)

## 2. The catch-up happens only from below

**Emerging economies converge fast.**
- A 1-point higher initial log productivity means 2.2 points lower annual growth (β = −2.20\*\*\*).
- That is about 3 % of the gap closed per year (C1).
- The fastest are the Baltics, Romania, Poland and Bulgaria.

**The leaders do not converge among themselves.** Their slope is +1.51\*\*\*, but it is estimated on eight countries that start at almost the same level.

![Beta convergence](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig7_beta_convergence.png)

**Productivity follows several paths, not one.** The Phillips–Sul log-t test rejects convergence of all countries to a single EU path. It finds three clubs, plus Denmark, Sweden and Bulgaria converging to none of them (B4).

![Convergence clubs](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig2_convergence_clubs.png)

## 3. Productivity drives R&D more than R&D drives productivity

Dumitrescu–Hurlin panel Granger tests (B6):
- **Productivity → R&D:** significant (p = 0.010 with one lag, 0.020 with two).
- **R&D → productivity:** not significant (p = 0.37 / 0.97).

Countries raise R&D *after* they get richer. So a fixed-effects regression of productivity on same-year R&D is biased: it returns a **negative** R&D coefficient for the leaders (A3, B3). That coefficient turns positive after one to three years, a **J-curve** (A3b).

![J-curve](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig3_jcurve_lags.png)

## 4. Once the feedback is removed, R&D pays off, mainly in the leaders

Blundell–Bond **system GMM** treats R&D as endogenous and instruments it with its own lags (B13). Of twelve specifications, only two pass every diagnostic: AR(1) rejects, AR(2) does not, Hansen J is acceptable, and there are no more instruments than countries. In both, R&D is positive and significant:

| R&D measure | Short-run effect | Reading |
|---|---|---|
| log R&D knowledge stock per worker | β = 4.97\*\* | 10 % larger stock → ≈ 0.5 % higher output per worker |
| R&D % of GDP | β = 3.65\*\* | +0.1 pp of GDP → ≈ 0.36 % higher output per worker |

By group, the effect is **positive for the Innovative economies** (stock p = 0.046, R&D % GDP p = 0.001) and smaller for the Emerging ones. The difference is not significant. Fixed effects with the same regressors, which ignore the feedback, give small, insignificant effects.

![GMM specifications](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig5_gmm_specifications.png)

## 5. …but the evidence is fragile

- **Instrument choice matters.** With deeper instrument lags the GMM effect loses significance, and the AR(1) test stops rejecting, a sign of weak instruments. Only the shallow-lag specification is both valid and significant.
- **Long-run effects are imprecise.** Productivity is highly persistent (ρ ≈ 0.9), so long-run multipliers carry wide intervals.
- **The frontier pattern is suggestive, not firm.** Double ML finds the R&D effect shrinking with distance to the technology frontier in 7 of 8 sample/learner combinations. The slope is significant in 6 of 8, but not in the default sample with the Random Forest learner (B12).

![DML sensitivity](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig6_dml_sensitivity.png)

## 6. EU accession cannot be separated from the catch-up already under way

A naive Callaway–Sant'Anna event study finds output per worker about 14 % higher after accession. But the accession countries were already diverging upward beforehand: the pre-trends are significant. After adjusting for them, the effect is between −4.5 % and +3.4 % (B9).

![Event study](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig4_event_study.png)

## 7. R&D tax incentives raise R&D spending a little, not productivity (yet)

**How the reforms are dated.** The OECD's implied R&D tax subsidy rate (1 − B-index) dates each country's generosity reforms without any hand-coding: a reform is the first year the large-firm subsidy rises by at least 5 pp and stays up. That finds **17 reforms**, and the dates match well-documented ones, such as Czechia 2005, France 2004, Lithuania 2008, Slovakia 2015, Poland 2016 and Germany 2020 (D2).

**Unlike EU accession, these reforms pass the parallel-trends check** (pre-trend p = 0.38–0.91). A staggered event study finds:
- **R&D intensity:** about +3–5 % after a reform, growing to +6–10 % after five years. The intervals are wide, so it is not significant.
- **Output per worker:** no response within five years.

![R&D tax reforms](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/results/figures/fig8_tax_reforms.png)

**Continuous subsidy changes behave differently.** They are followed by *lower* R&D intensity (D3), which suggests governments raise support when R&D is weak. That policy endogeneity is why the event study, with its testable pre-trends, is the preferred design.

**The fiscal result is not causal.** Output rises after the government balance improves, while employment does not move (D4). But the headline balance improves automatically in booms, so this is mostly the business cycle. A causal test needs the cyclically adjusted balance.

## 8. Other results

- **Savings** are the most consistent productivity driver on the thesis panel (A2, A3), and weaker on the rebuilt panel (B3).
- **Hausman** prefers fixed effects only at the 10 % level (p = 0.099; A2).
- **Machine learning.** With honest country-held-out cross-validation, out-of-sample R² is about 0.05, and forecasting later years does worse than the mean (A6). Productivity growth is hard to predict, so treat ML scenario tools with caution.

## How to write this up

1. **Reframe the GERD paradox.** Present it as a matter of **timing and reverse causality**, not only aggregation: R&D follows income, and its payoff arrives with a lag and mainly where absorptive capacity is high.
2. **Show the estimates with their diagnostics.** Report the GMM estimates together with the full specification grid (Fig. 5, `results/tables/tab_gmm_grid.tex`).
3. **Cite the data audit.** The submitted panel's `Human Capital Proxy` is misaligned (see [[Data and Audit]]).
4. **Say which convergence holds.** It is established for the emerging economies and within clubs, not for the leaders or for the EU as a whole.
