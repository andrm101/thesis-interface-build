"""
tabs/tab_stats.py  —  Exploratory data analysis: descriptive statistics,
statistical test batteries and EDA figures.

Tests (tables, FDR-adjusted where many variables are tested at once — see
eda.py): normality, group differences (Innovative vs Emerging or K-Means
clusters), country heterogeneity, trends, pooled/between/within
correlations, Pesaran CD, variance decomposition, structural breaks and
lead-lag correlations.

Figures: correlation heatmap, distributions, Q-Q plots, trajectories with
group means, box plots by country, country × year heatmap, z-score map,
scatter matrix, X-vs-Y by group, lead-lag correlogram, between/within
variance bars.

Mixin: StatsTabMixin
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats as sps

import eda
import outliers
import theme
from constants import INNOVATIVE_CLUSTER, stars
from helpers import make_text, write, clear_txt, embed_figure

KEY_VARS = ["Y by L", "PIB towards research", "Labor in research",
            "Human Capital Proxy", "Savings Percentage", "Patents per capita"]
GROUP_SOURCES = ("A-priori thesis clusters", "K-Means clusters (Tab 5)")


class StatsTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — EXPLORATORY DATA ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_stats(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=12, pady=6)

        opts = ttk.Frame(ctrl)
        opts.pack(fill=tk.X, pady=2)
        self.st_var    = tk.StringVar(value="PIB towards research")
        self.st_y      = tk.StringVar(value="Y by L")
        self.st_group  = tk.StringVar(value=GROUP_SOURCES[0])
        self.st_maxlag = tk.IntVar(value=5)
        ttk.Label(opts, text="Variable / X:").pack(side=tk.LEFT)
        self.st_var_cb = ttk.Combobox(opts, textvariable=self.st_var,
                                      state="readonly", width=24)
        self.st_var_cb.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(opts, text="Outcome Y:").pack(side=tk.LEFT)
        self.st_y_cb = ttk.Combobox(opts, textvariable=self.st_y,
                                    state="readonly", width=18)
        self.st_y_cb.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(opts, text="Groups:").pack(side=tk.LEFT)
        ttk.Combobox(opts, textvariable=self.st_group, values=GROUP_SOURCES,
                     state="readonly", width=24).pack(side=tk.LEFT,
                                                      padx=(4, 12))
        ttk.Label(opts, text="Max lag:").pack(side=tk.LEFT)
        sp = tk.Spinbox(opts, from_=1, to=10, textvariable=self.st_maxlag,
                        width=4, bg=theme.WBG, fg=theme.FG,
                        buttonbackground=theme.WBG,
                        insertbackground=theme.FG)
        sp.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(sp)

        tests = [
            ("Summary",            self._run_summary),
            ("Normality",          self._run_normality),
            ("Group Differences",  self._run_group_tests),
            ("Country Effects",    self._run_country_tests),
            ("Trends",             self._run_trend_tests),
            ("Correlations",       self._run_corr_tests),
            ("Cross-sec. Dep.",    self._run_cd_tests),
            ("Variance Decomp.",   self._run_vardecomp),
            ("Structural Breaks",  self._run_break_tests),
            ("Lead-Lag",           self._run_leadlag_table),
        ]
        figs = [
            ("Correlation Heatmap", self._run_corr),
            ("Distributions",       self._run_dist),
            ("Q-Q Plots",           self._plot_qq),
            ("Trajectories",        self._plot_trajectories),
            ("Box by Country",      self._plot_box_country),
            ("Country × Year",      self._plot_country_year),
            ("Z-score Map",         self._plot_zmap),
            ("Scatter Matrix",      self._plot_pairs),
            ("X vs Y by Group",     self._plot_xy_groups),
            ("Lead-Lag Plot",       self._plot_leadlag),
            ("Between / Within",    self._plot_vardecomp),
        ]
        for title, items, extra in (("Tests", tests, True),
                                    ("Figures", figs, False)):
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{title}:", width=8).pack(side=tk.LEFT)
            for label, cmd in items:
                ttk.Button(row, text=label, command=cmd).pack(
                    side=tk.LEFT, padx=2)
            if extra:
                ttk.Button(row, text="▶ Run All Tests",
                           style="Accent.TButton",
                           command=self._run_all_eda_tests).pack(
                    side=tk.LEFT, padx=8)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.stats_txt        = make_text(left, height=30, copy_root=self.root)
        self.stats_plot_frame = right

    def _refresh_stats_vars(self):
        if self.df is None:
            return
        cols = eda.numeric_vars(self.df)
        for cb, var in ((self.st_var_cb, self.st_var),
                        (self.st_y_cb, self.st_y)):
            cb["values"] = cols
            if var.get() not in cols and cols:
                var.set(cols[0])

    # ── Shared helpers ────────────────────────────────────────────────────────
    def _eda_ready(self) -> bool:
        if not self._check_data():
            return False
        self._refresh_stats_vars()
        return True

    def _eda_groups(self) -> pd.Series:
        """Country → group label, from K-Means (if chosen and run) or the
        a-priori thesis clusters."""
        countries = self.df["Country"].unique()
        if (self.st_group.get() == GROUP_SOURCES[1]
                and self.clusters is not None):
            cl = self.clusters.set_index("Country")["Cluster"]
            return pd.Series({c: f"Cluster {cl[c]}" for c in countries
                              if c in cl.index})
        return pd.Series({c: ("Innovative" if c.lower() in INNOVATIVE_CLUSTER
                              else "Emerging") for c in countries})

    def _key_vars(self) -> list[str]:
        cols = [c for c in KEY_VARS if c in self.df.columns]
        return cols or eda.numeric_vars(self.df)[:6]

    def _new_fig(self, w=11, h=8):
        for wdg in self.stats_plot_frame.winfo_children():
            wdg.destroy()
        return Figure(figsize=(w, h), facecolor=theme.BG)

    def _show_fig(self, fig):
        fig.tight_layout()
        embed_figure(fig, self.stats_plot_frame)

    @staticmethod
    def _fmt_table(df: pd.DataFrame, title: str, note: str = "",
                   first_w: int = 24) -> str:
        """Fixed-width text table; p-value columns get significance stars."""
        if df is None or df.empty:
            return f"\n{title}\n  (not enough data)\n"
        cols = list(df.columns)
        widths = [max(first_w, len(str(cols[0])))] + \
                 [max(9, len(str(c)) + 1) for c in cols[1:]]
        head = "".join(f"{str(c):<{w}}" if i == 0 else f"{str(c):>{w}}"
                       for i, (c, w) in enumerate(zip(cols, widths)))
        out = [f"\n{title}\n{'=' * len(head)}\n", head + "\n",
               "─" * len(head) + "\n"]
        for _, row in df.iterrows():
            cells = []
            for i, (c, w) in enumerate(zip(cols, widths)):
                v = row[c]
                if i == 0:
                    cells.append(f"{str(v)[:w - 1]:<{w}}")
                elif isinstance(v, (float, np.floating)):
                    s = "nan" if not np.isfinite(v) else (
                        f"{v:.3f}" if abs(v) < 1e5 else f"{v:.2e}")
                    if str(c).endswith(("_p", "p", "_fdr")) and np.isfinite(v):
                        s = f"{v:.3f}{stars(v):<3}"
                    cells.append(f"{s:>{w}}")
                else:
                    cells.append(f"{str(v)[:w - 1]:>{w}}")
            out.append("".join(cells) + "\n")
        if note:
            out.append(note.rstrip() + "\n")
        return "".join(out)

    def _emit(self, text: str, clear: bool = True):
        if clear:
            clear_txt(self.stats_txt)
        write(self.stats_txt, text)

    # ══════════════════════════════════════════════════════════════════════════
    # TESTS
    # ══════════════════════════════════════════════════════════════════════════
    def _run_summary(self, clear=True):
        if not self._eda_ready():
            return
        df   = self.df[eda.numeric_vars(self.df)]
        desc = df.describe().T
        tbl  = pd.DataFrame({
            "Variable": desc.index, "N": desc["count"].astype(int),
            "Mean": desc["mean"], "Std": desc["std"], "Min": desc["min"],
            "Median": desc["50%"], "Max": desc["max"],
            "CV": desc["std"] / desc["mean"].abs(),
            "Missing%": df.isna().mean().values * 100})
        self._emit(self._fmt_table(tbl, "DESCRIPTIVE STATISTICS"), clear)

    def _run_normality(self, clear=True):
        if not self._eda_ready():
            return
        t = eda.normality_tests(self.df)
        t = t[["Variable", "N", "Skew", "Kurtosis", "SW_p", "JB_p", "K2_p",
               "JB_p_fdr"]] if len(t) else t
        self._emit(self._fmt_table(
            t, "NORMALITY — Shapiro-Wilk, Jarque-Bera, D'Agostino K²",
            "  H0: normal. Kurtosis is excess kurtosis. *_fdr = "
            "Benjamini-Hochberg adjusted.\n"
            "  Non-normal levels motivate log transforms / robust SEs."),
            clear)

    def _run_group_tests(self, clear=True):
        if not self._eda_ready():
            return
        g = self._eda_groups()
        t = eda.group_difference_tests(self.df, g)
        if len(t):
            t = t[["Variable", "Mean_A", "Mean_B", "Cohen_d", "Welch_p",
                   "MWU_p", "KS_p", "Levene_p", "MWU_p_fdr"]]
        labels = list(pd.unique(g.dropna()))[:2]
        self._emit(self._fmt_table(
            t, f"GROUP DIFFERENCES — A = {labels[0] if labels else '?'}, "
               f"B = {labels[1] if len(labels) > 1 else '?'}",
            "  Welch t (means), Mann-Whitney U (location), Kolmogorov-Smirnov"
            " (whole distribution),\n  Levene (variances). |d| ≥ 0.2 small, "
            "0.5 medium, 0.8 large.\n  Observations are country-years, so "
            "p-values overstate independence — read with d."),
            clear)

    def _run_country_tests(self, clear=True):
        if not self._eda_ready():
            return
        t = eda.country_effect_tests(self.df)
        self._emit(self._fmt_table(
            t, "COUNTRY HETEROGENEITY — one-way ANOVA & Kruskal-Wallis",
            "  H0: all countries share the same mean / distribution.\n"
            "  Eta² = share of variance explained by country → large values"
            " justify entity fixed effects."), clear)

    def _run_trend_tests(self, clear=True):
        if not self._eda_ready():
            return
        t = eda.trend_tests(self.df)
        self._emit(self._fmt_table(
            t, "TRENDS — Mann-Kendall on cross-country yearly mean",
            "  Countries_up / _down: countries with a significant (5 %) "
            "individual Mann-Kendall trend.\n  Persistent trends point to "
            "non-stationarity — confirm in Tab 10 (Unit Roots)."), clear)

    def _run_corr_tests(self, clear=True):
        if not self._eda_ready():
            return
        y = self.st_y.get()
        t = eda.correlation_tests(self.df, y)
        if len(t):
            keep = ["Variable", "Pooled_r", "Pooled_p", "Between_r",
                    "Between_p", "Within_r", "Within_p", "Within_p_fdr"]
            t = t[[c for c in keep if c in t.columns]]
        self._emit(self._fmt_table(
            t, f"CORRELATIONS WITH {y} — pooled / between / within",
            "  Between = country means (cross-section); Within = deviations "
            "from country means\n  (what fixed effects identify). A sign "
            "flip between them is a Simpson's-paradox warning\n  — the "
            "mechanism behind the aggregate GERD paradox."), clear)

    def _run_cd_tests(self, clear=True):
        if not self._eda_ready():
            return
        t = eda.cd_tests(self.df)
        self._emit(self._fmt_table(
            t, "CROSS-SECTIONAL DEPENDENCE — Pesaran (2004) CD",
            "  H0: no cross-sectional dependence (on country-demeaned "
            "series).\n  Rejection → common shocks; prefer Driscoll-Kraay SE "
            "(Tab 12) and CIPS-type unit-root tests."), clear)

    def _run_vardecomp(self, clear=True):
        if not self._eda_ready():
            return
        t = eda.variance_decomposition(self.df)
        self._emit(self._fmt_table(
            t, "VARIANCE DECOMPOSITION — between vs. within countries",
            "  Low Within_share → little time variation left for fixed "
            "effects to exploit\n  (FE estimates of that regressor will be "
            "imprecise)."), clear)

    def _run_break_tests(self, clear=True):
        if not self._eda_ready():
            return
        t = eda.structural_break_tests(self.df)
        if len(t):
            t = t[["Variable"] + [c for c in t.columns if c.startswith("p_")]]
        self._emit(self._fmt_table(
            t, "STRUCTURAL BREAKS — Chow test on yearly-mean trend",
            "  Candidate breaks: 2004 enlargement, 2008-09 GFC, 2013 Croatia "
            "accession / post-crisis,\n  2020 COVID. H0: no break in the "
            "linear trend at that year."), clear)

    def _run_leadlag_table(self, clear=True):
        if not self._eda_ready():
            return
        x, y = self.st_var.get(), self.st_y.get()
        t = eda.lead_lag_correlation(self.df, x, y, int(self.st_maxlag.get()))
        self._emit(self._fmt_table(
            t, f"LEAD-LAG — corr({x}[t−k], {y}[t]), within-country",
            "  k > 0: X leads Y (consistent with R&D → productivity with a "
            "delay);\n  k < 0: Y leads X (reverse causality / pro-cyclical "
            "R&D). band = ±1.96/√n.", first_w=6), clear)

    def _run_all_eda_tests(self):
        if not self._eda_ready():
            return
        clear_txt(self.stats_txt)
        for f in (self._run_summary, self._run_normality,
                  self._run_group_tests, self._run_country_tests,
                  self._run_trend_tests, self._run_corr_tests,
                  self._run_cd_tests, self._run_vardecomp,
                  self._run_break_tests, self._run_leadlag_table):
            try:
                f(clear=False)
            except Exception as exc:
                write(self.stats_txt, f"\n[{f.__name__}] failed: {exc}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    def _run_corr(self):
        if not self._eda_ready():
            return
        fig  = self._new_fig(10, 8)
        corr = self.df[eda.numeric_vars(self.df)].corr()
        ax   = fig.add_subplot(111)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                    fmt=".2f", linewidths=0.4, ax=ax,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix (lower triangle)",
                     color=theme.FG, pad=12)
        ax.tick_params(colors=theme.FG, labelsize=8)
        self._show_fig(fig)

    def _run_dist(self):
        if not self._eda_ready():
            return
        cols = self._key_vars()
        g    = self.df["Country"].map(self._eda_groups())
        pal  = theme.cluster_palette()
        fig  = self._new_fig(12, 8)
        for i, col in enumerate(cols, 1):
            ax = fig.add_subplot(2, 3, i)
            for j, lab in enumerate(pd.unique(g.dropna())):
                data = self.df.loc[g == lab, col].dropna()
                ax.hist(data, bins=20, color=pal[j % len(pal)], alpha=0.55,
                        edgecolor=theme.DARK, label=lab, density=True)
                if data.nunique() > 1:
                    xs = np.linspace(data.min(), data.max(), 200)
                    ax.plot(xs, sps.gaussian_kde(data)(xs),
                            color=pal[j % len(pal)], lw=1.5)
            ax.set_title(col, color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=7)
            if i == 1:
                ax.legend(fontsize=7)
        fig.suptitle("Distributions by group (histogram + KDE)",
                     color=theme.FG, fontsize=12)
        self._show_fig(fig)

    def _plot_qq(self):
        if not self._eda_ready():
            return
        fig = self._new_fig(12, 8)
        for i, col in enumerate(self._key_vars(), 1):
            ax = fig.add_subplot(2, 3, i)
            x  = self.df[col].dropna()
            (osm, osr), (slope, icpt, r) = sps.probplot(x, dist="norm")
            ax.scatter(osm, osr, s=8, color=theme.BLUE, alpha=0.7)
            ax.plot(osm, slope * osm + icpt, color=theme.RED, lw=1.2)
            ax.set_title(f"{col}  (r={r:.3f})", color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=7)
        fig.suptitle("Normal Q-Q plots", color=theme.FG, fontsize=12)
        self._show_fig(fig)

    def _plot_trajectories(self):
        if not self._eda_ready():
            return
        g   = self._eda_groups()
        pal = theme.cluster_palette()
        fig = self._new_fig(12, 8)
        for i, col in enumerate(self._key_vars(), 1):
            ax = fig.add_subplot(2, 3, i)
            for c, sub in self.df.groupby("Country"):
                ax.plot(sub["Year"], sub[col], color=theme.GRAY, lw=0.5,
                        alpha=0.35)
            for j, lab in enumerate(pd.unique(g.dropna())):
                sub = self.df[self.df["Country"].map(g) == lab]
                q = sub.groupby("Year")[col].quantile([0.25, 0.5, 0.75]
                                                      ).unstack()
                colr = pal[j % len(pal)]
                ax.fill_between(q.index, q[0.25], q[0.75], color=colr,
                                alpha=0.18)
                ax.plot(q.index, q[0.5], color=colr, lw=2, label=lab)
            ax.set_title(col, color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=7)
            if i == 1:
                ax.legend(fontsize=7)
        fig.suptitle("Country trajectories (grey) · group median ± IQR",
                     color=theme.FG, fontsize=12)
        self._show_fig(fig)

    def _plot_box_country(self):
        if not self._eda_ready():
            return
        col   = self.st_var.get()
        g     = self._eda_groups()
        pal   = theme.cluster_palette()
        order = (self.df.groupby("Country")[col].median()
                 .sort_values().index.tolist())
        labs  = list(pd.unique(g.dropna()))
        fig   = self._new_fig(12, 7)
        ax    = fig.add_subplot(111)
        data  = [self.df.loc[self.df["Country"] == c, col].dropna()
                 for c in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        ax.set_xticks(range(1, len(order) + 1), order)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(pal[labs.index(g[c]) % len(pal)]
                                if c in g.index else theme.GRAY)
            patch.set_alpha(0.7)
        ax.tick_params(axis="x", rotation=60, labelsize=8, colors=theme.FG)
        ax.set_title(f"{col} by country (sorted by median; colour = group)",
                     color=theme.FG)
        self._show_fig(fig)

    def _plot_country_year(self):
        if not self._eda_ready():
            return
        col  = self.st_var.get()
        wide = self.df.pivot_table(index="Country", columns="Year",
                                   values=col)
        wide = wide.loc[wide.mean(axis=1).sort_values().index]
        fig  = self._new_fig(12, 8)
        ax   = fig.add_subplot(111)
        sns.heatmap(wide, cmap="viridis", ax=ax, linewidths=0.2,
                    cbar_kws={"label": col})
        ax.set_title(f"{col}: country × year (rows sorted by mean)",
                     color=theme.FG)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(colors=theme.FG, labelsize=7)
        self._show_fig(fig)

    def _plot_zmap(self):
        if not self._eda_ready():
            return
        cols = outliers.default_screen_vars(self.df)
        z    = outliers.country_zscores(self.df, cols)
        fig  = self._new_fig(10, 8)
        ax   = fig.add_subplot(111)
        sns.heatmap(z, cmap="RdBu_r", center=0, vmin=-3.5, vmax=3.5,
                    annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax,
                    linewidths=0.3, cbar_kws={"label": "z"})
        ax.set_title("Country-mean z-scores (current filtered sample)",
                     color=theme.FG)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(colors=theme.FG, labelsize=8)
        self._show_fig(fig)

    def _plot_pairs(self):
        if not self._eda_ready():
            return
        cols = self._key_vars()[:5]
        g    = self.df["Country"].map(self._eda_groups())
        labs = list(pd.unique(g.dropna()))
        pal  = theme.cluster_palette()
        n    = len(cols)
        fig  = self._new_fig(12, 10)
        for i, yv in enumerate(cols):
            for j, xv in enumerate(cols):
                ax = fig.add_subplot(n, n, i * n + j + 1)
                for k, lab in enumerate(labs):
                    sub = self.df[g == lab]
                    if i == j:
                        ax.hist(sub[xv].dropna(), bins=15, alpha=0.55,
                                color=pal[k % len(pal)])
                    else:
                        ax.scatter(sub[xv], sub[yv], s=4, alpha=0.5,
                                   color=pal[k % len(pal)], label=lab)
                ax.tick_params(labelsize=5, colors=theme.FG)
                if i == n - 1:
                    ax.set_xlabel(xv, fontsize=6, color=theme.FG)
                if j == 0:
                    ax.set_ylabel(yv, fontsize=6, color=theme.FG)
        fig.suptitle("Scatter matrix (colour = group)", color=theme.FG)
        self._show_fig(fig)

    def _plot_xy_groups(self):
        if not self._eda_ready():
            return
        x, y = self.st_var.get(), self.st_y.get()
        g    = self.df["Country"].map(self._eda_groups())
        labs = list(pd.unique(g.dropna()))
        pal  = theme.cluster_palette()
        d    = self.df[["Country", x, y]].assign(G=g).dropna()
        dm   = d.copy()
        dm[[x, y]] = d[[x, y]] - d.groupby("Country")[[x, y]].transform("mean")
        fig  = self._new_fig(12, 6)
        for pos, (data, ttl) in enumerate(((d, "Pooled levels"),
                                           (dm, "Within-country "
                                                "(demeaned, FE view)")), 1):
            ax = fig.add_subplot(1, 2, pos)
            for k, lab in enumerate(labs):
                sub = data[data["G"] == lab]
                colr = pal[k % len(pal)]
                ax.scatter(sub[x], sub[y], s=10, alpha=0.5, color=colr)
                if len(sub) > 2 and sub[x].nunique() > 1:
                    lr = sps.linregress(sub[x], sub[y])
                    xs = np.linspace(sub[x].min(), sub[x].max(), 50)
                    ax.plot(xs, lr.intercept + lr.slope * xs, color=colr,
                            lw=2, label=f"{lab}: β={lr.slope:.3f} "
                                        f"(p={lr.pvalue:.3f})")
            ax.set_xlabel(x); ax.set_ylabel(y)
            ax.set_title(ttl, color=theme.FG)
            ax.legend(fontsize=7)
        fig.suptitle(f"{y} vs {x} by group", color=theme.FG)
        self._show_fig(fig)

    def _plot_leadlag(self):
        if not self._eda_ready():
            return
        x, y = self.st_var.get(), self.st_y.get()
        t = eda.lead_lag_correlation(self.df, x, y, int(self.st_maxlag.get()))
        if t.empty:
            messagebox.showinfo("Lead-Lag", "Not enough data.")
            return
        fig = self._new_fig(10, 5.5)
        ax  = fig.add_subplot(111)
        cols = [theme.TEAL if abs(r) > b else theme.GRAY
                for r, b in zip(t["r"], t["band"])]
        ax.bar(t["Lag"], t["r"], color=cols)
        ax.plot(t["Lag"], t["band"], "--", color=theme.RED, lw=1)
        ax.plot(t["Lag"], -t["band"], "--", color=theme.RED, lw=1)
        ax.axhline(0, color=theme.FG, lw=0.6)
        ax.axvline(0, color=theme.GRAY, lw=0.6, ls=":")
        ax.set_xlabel(f"k  (k > 0: {x} leads {y})")
        ax.set_ylabel("within-country correlation")
        ax.set_title(f"Lead-lag correlogram: {x}[t−k] vs {y}[t]",
                     color=theme.FG)
        self._show_fig(fig)

    def _plot_vardecomp(self):
        if not self._eda_ready():
            return
        t = eda.variance_decomposition(self.df).sort_values("Within_share")
        fig = self._new_fig(10, 7)
        ax  = fig.add_subplot(111)
        ax.barh(t["Variable"], t["Between_share"], color=theme.BLUE,
                label="Between countries")
        ax.barh(t["Variable"], t["Within_share"], left=t["Between_share"],
                color=theme.AMBER, label="Within countries (over time)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of total variance")
        ax.set_title("Variance decomposition — what fixed effects can "
                     "identify", color=theme.FG)
        ax.legend(loc="lower right", fontsize=8)
        ax.tick_params(labelsize=8)
        self._show_fig(fig)
