# Augmented Solow · R&D Heterogeneity Lab

> **Thesis interface for:**
> *An empirical investigation into the heterogeneous impact of R&D
> investment on economic productivity across 25 EU member states over a
> 25-year panel (1998–2023).*
>
> **Shipped sample:** the `panel_data.xlsx` bundled here covers **23 countries, 2000–2023** (24 years, 552 rows) — Austria, Cyprus, and Ireland from the thesis's original EU-25/1998 scope are not present in this shipped subsample.

An interactive desktop application (Tkinter + matplotlib) that reproduces
the empirical workflow of the thesis end-to-end. It loads a World-Bank
panel of EU economies, lets you build and estimate an **augmented
Solow model** enriched with R&D and patent indicators, runs a **K-Means
typology** to separate Innovative leaders from Emerging adopters, and
then estimates **panel FE / RE** models inside each cluster — validating
the specification with the **Hausman test** and confirming stationarity
with **ADF / PP / Im-Pesaran-Shin** unit-root tests.

---

## Thesis in one paragraph

The study extends the Solow-Swan growth framework with R&D expenditure
(GERD) and patent activity as innovation proxies. K-Means clustering
partitions the 25 EU economies into two structurally distinct
typologies — Nordic / Central-European innovation leaders and
Eastern / Southern emerging adopters. Panel fixed-effects and
random-effects models are estimated separately within each cluster, with
Hausman tests guiding the specification choice. The main finding
resolves the **GERD paradox**: an aggregate negative relationship
between R&D spending and productivity disappears once the sample is
split, revealing strong positive effects in innovator economies that
were masked by the low absorptive capacity of emerging ones. **Savings
rate** is the most robust productivity driver across all specifications,
and **conditional β-convergence** is confirmed in both clusters through
significantly negative initial-GDP coefficients.

---

## Architecture

```mermaid
flowchart TD
    WB["World Bank panel<br/>23 EU countries, 2000-2023"] --> Load["main.py — panel_data.xlsx load"]
    Load --> Solow["Augmented Solow model<br/>+ R&D / patent indicators"]
    Solow --> KMeans["K-Means typology<br/>(Innovative leaders / Emerging adopters)"]
    KMeans --> FERE["Panel FE / RE per cluster"]
    FERE --> Hausman["Hausman test"]
    FERE --> UnitRoot["ADF / PP / Im-Pesaran-Shin"]
    Hausman --> UI["Tkinter + matplotlib<br/>desktop interface (tabs/)"]
    UnitRoot --> UI
```

## Quick start

```bash
# 1. Clone or copy the folder, then install deps (Python ≥ 3.10)
pip install pandas numpy matplotlib seaborn scikit-learn \
            statsmodels linearmodels openpyxl

# 2. Launch the GUI
python main.py
```

On Windows the app opens maximised and applies per-monitor DPI
awareness. A dark "Palantir" palette is the default; toggle to the
light "Claude" palette from the top-right corner for paper-ready
screenshots.

---

## How to run the analysis for thesis-matching results

The interface is organised as a linear 13-tab pipeline. Following the
tabs top-to-bottom reproduces the empirical workflow of the thesis.

### 1 · Data
1. Click **Browse…** and pick `panel_data.xlsx` (shipped alongside the
   code; 552 rows × 16 cols, long-form country × year).
2. Leave the year range at **2000 – 2023**.
3. Click **EU-25** (or **Innovative** / **Emerging** if you already want
   to focus on one cluster).
4. Hit **Apply Filters**. The status bar should read *"552 obs · 23
   countries · 24 years"*.
5. Optional but recommended: click **Panel Structure Report** — it
   prints whether the panel is balanced and a per-variable missingness
   table. Expect a balanced 23 × 24 = 552-cell grid.
6. In **Data Transformation Tools**, pre-build:
   - `Y by L → Log-Level` (for productivity).
   - `PIB towards research → Log-Level`.
   - `Patents per capita → Growth Rate (%)` if you want a "patent
     intensity" flow variable.

### 2 · Statistics
Run descriptive stats and correlation heatmaps by cluster. Confirms that
R&D intensity, patents, and savings rate all differ significantly
between Innovative and Emerging economies.

### 3 · Panel FE / RE  *(core estimation)*
1. **Dependent variable**: `Y by L` (output per worker).
2. **Regressors** (default pre-selected): Savings Percentage, Human
   Capital Proxy, Labor in research, PIB towards research, Patents per
   capita, Labor not in research.
3. **Model**: start with *Fixed Effects (Entity + Time)* — the thesis'
   preferred within estimator.
4. Keep **Log-transform** ON and **Clustered SE (entity)** ON.
5. (Optional) set *Lag depth for regressors* to `1` and click **Build
   Lag Variables** to reproduce the dynamic specification.
6. Click **Run Model**. Cross-check with **Hausman Test (FE vs RE)** —
   you should reject H₀ ⇒ prefer Fixed Effects.
7. **Plot Significant Variables** renders the forest plot used in the
   thesis' Figure 4.

Repeat separately for the **Innovative** and **Emerging** filters
(set in Tab 1) to reproduce the cluster-wise results that resolve the
GERD paradox.

### 4 · Convergence
Run a cross-sectional OLS of average productivity growth on
**initial log-GDP** to test **β-convergence**. Do it once for each
cluster — both slopes should be negative and significant, confirming
conditional convergence.

### 5 · Clustering  *(the typology)*
1. Keep the default variables selected: **PIB towards research**,
   **Labor in research**, **Patents per capita**, **Human Capital
   Proxy**.
2. **K = 2**.
3. Click **Elbow Plot** — the annotated bend at K = 2 confirms the
   choice.
4. Click **Run Clustering**. The text panel lists cluster membership
   and auto-labels the high-R&D group as **Innovative** and the other
   as **Emerging**.
5. Click **PCA Biplot** for the two-dimensional visualisation with
   loading arrows — this mirrors the thesis' country-typology figure.

### 6 · ML Models
Train supervised models (Random Forest, Gradient Boosting, etc.) using
the same productivity target. The ML results are benchmarks for the
econometric models — they should match the sign and rank of the key
drivers identified by the panel FE specification.

### 7 · Scenarios
Counterfactuals: move a country's R&D intensity to the cluster mean and
re-predict productivity. Quantifies the absorptive-capacity channel
behind the GERD paradox.

### 8 · Compare
Side-by-side regression table (Pooled / FE / RE / FE+Time) — the
Table-2 equivalent of the thesis.

### 9 · Diagnostics
Residual normality, heteroskedasticity (White test), serial correlation,
and cross-sectional dependence checks.

### 10 · Unit Roots
**ADF**, **PP**, and **Im-Pesaran-Shin** panel unit-root tests. Reject
the null of a unit root for log-differenced variables before using them
in the panel specification.

### 11 · VAR / IRF
Vector autoregression and impulse-response functions — R&D shock → TFP
and Output responses. Supplementary to the main specification.

### 12 · Advanced
Non-linear and interaction specifications (R&D × Human Capital, etc.).

### 13 · Report
Export any tab's text panel to `.txt` and summarise model fits in a
single consolidated report.

---

## Matching the thesis' reported results

| Thesis finding | Where to reproduce | Expected sign |
|---|---|---|
| K-Means identifies 2 typologies | Tab 5 · Clustering (K=2) | Innovative / Emerging split matching `constants.INNOVATIVE_CLUSTER` and `EMERGING_CLUSTER`. |
| GERD paradox: negative aggregate, positive in leaders | Tab 3 · Panel FE with whole EU-25 then Innovative only | EU-25 coef on R&D ≤ 0; Innovative coef > 0 at 5 %. |
| Savings rate is the strongest driver | Tab 3 · every FE/RE run | Positive and *** across all specs. |
| Conditional convergence | Tab 4 · Convergence (per cluster) | Negative, significant β on initial log-GDP. |
| Hausman prefers FE | Tab 3 · Hausman button | χ² p < 0.05 ⇒ reject RE. |
| Variables are I(1) in levels, I(0) in log-diffs | Tab 10 · Unit Roots | ADF/PP/IPS reject stationarity on levels, fail to reject on log-differences. |

---

## Expected dataset schema

The app was built around the shipped `panel_data.xlsx`, but it accepts
any long-form panel with at least the columns below:

| Column | Role | Notes |
|---|---|---|
| `Country` | Entity | String, one per country. |
| `Year` | Time | Integer. |
| `Y by L` | Dependent (productivity) | Output per worker. |
| `Savings Percentage` | Key driver | Gross-savings share of GDP. |
| `Human Capital Proxy` | Key driver | Any schooling/education index. |
| `Labor in research` | R&D labour | Researchers, FTE. |
| `Labor not in research` | Residual labour | Complement to the above. |
| `PIB towards research` | GERD intensity | R&D as % of GDP. |
| `Patents per capita` | Innovation output | Patent applications normalised. |
| `Patents`, `Patents per hour` | Optional extra innovation proxies | |
| `Population`, `Labor` | Scale variables | |
| `TFP Growth Rate`, `A` | Total factor productivity | Computed exogenously. |

Other columns are tolerated — the app infers numeric variables
automatically and offers them in every dropdown.

---

## Project layout

```
Thesis Interface Build/
├── main.py                  # entry point — run this
├── app.py                   # ThesisApp shell (multiple-inheritance of mixins)
├── theme.py                 # palettes, ttk styles, matplotlib defaults
├── constants.py             # EU groupings + thesis clusters + helpers
├── helpers.py               # shared UI + plot embedding helpers
├── panel_data.xlsx          # shipped sample: 23 countries, 2000-2023
├── tabs/
│   ├── tab_data.py          # 1 · Data         — loading, filtering, transforms
│   ├── tab_stats.py         # 2 · Statistics   — descriptives + correlations
│   ├── tab_panel.py         # 3 · Panel FE/RE  — core estimation + Hausman
│   ├── tab_convergence.py   # 4 · Convergence  — cross-sectional β-convergence
│   ├── tab_clustering.py    # 5 · Clustering   — K-Means typology + PCA biplot
│   ├── tab_ml.py            # 6 · ML Models    — RF/GB/XGB benchmarks
│   ├── tab_scenarios.py     # 7 · Scenarios    — counterfactual R&D shocks
│   ├── tab_compare.py       # 8 · Compare      — side-by-side regression table
│   ├── tab_diagnostics.py   # 9 · Diagnostics  — White, BG, CD tests
│   ├── tab_unitroot.py      # 10 · Unit Roots  — ADF, PP, Im-Pesaran-Shin
│   ├── tab_var.py           # 11 · VAR / IRF   — impulse responses
│   ├── tab_advanced.py      # 12 · Advanced    — interactions, non-linearities
│   └── tab_report.py        # 13 · Report      — consolidated export
└── README.md                # this file
```

---

## Methods checklist (from the thesis abstract)

- [x] Augmented Solow model (GERD + patents)
- [x] K-Means clustering (K = 2)
- [x] Panel Fixed Effects / Random Effects
- [x] Hausman specification test
- [x] Unit root testing — ADF, PP, Im-Pesaran-Shin
- [x] Cross-sectional OLS (β-convergence)
- [x] White heteroscedasticity test

Every one of the above is exposed as a button in the GUI.

---

## Credits & licence

Built as a companion tool for Andrei's master-thesis on R&D
heterogeneity in EU growth. MIT-style reuse welcome — drop a link back
to the repo if you build on it.
