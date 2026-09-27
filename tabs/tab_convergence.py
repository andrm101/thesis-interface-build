"""
tabs/tab_convergence.py  —  β-convergence (absolute & conditional) and σ-convergence.
Mixin: ConvergenceTabMixin
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from matplotlib.figure import Figure

import clubs
import theme
from constants import stars, WESTERN_EU, EASTERN_EU
from helpers import make_text, write, clear_txt, embed_figure


class ConvergenceTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — CONVERGENCE
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_conv(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Convergence Analysis", padding=10)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Income variable:").pack(side=tk.LEFT)
        self.conv_var = ttk.Combobox(row0, width=22, state="readonly")
        self.conv_var.pack(side=tk.LEFT, padx=6)
        ttk.Label(row0, text="  Group:").pack(side=tk.LEFT, padx=6)
        self.conv_group = ttk.Combobox(row0, width=18, state="readonly",
            values=["All Countries", "Western EU", "Eastern EU", "By Cluster"])
        self.conv_group.current(0); self.conv_group.pack(side=tk.LEFT, padx=4)

        cv_frm = ttk.LabelFrame(ctrl,
            text="Conditioning variables  "
                 "(Ctrl+click for conditional β-convergence)",
            padding=6)
        cv_frm.pack(fill=tk.X, pady=4)
        sb_cv = ttk.Scrollbar(cv_frm); sb_cv.pack(side=tk.RIGHT, fill=tk.Y)
        self.conv_ctrl_lb = tk.Listbox(
            cv_frm, selectmode=tk.MULTIPLE, height=3,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            font=("Segoe UI", 9), yscrollcommand=sb_cv.set,
            exportselection=False,
        )
        self.conv_ctrl_lb.pack(fill=tk.BOTH, expand=True)
        sb_cv.config(command=self.conv_ctrl_lb.yview)

        btn_row = ttk.Frame(ctrl); btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Absolute β-Convergence",
                   command=self._run_beta_conv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Conditional β-Convergence",
                   style="Accent.TButton",
                   command=self._run_conditional_beta).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="σ-Convergence",
                   command=self._run_sigma_conv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Both (side by side)",
                   command=self._run_both_conv).pack(side=tk.LEFT, padx=4)

        ps = ttk.Frame(ctrl); ps.pack(fill=tk.X, pady=4)
        ttk.Label(ps, text="Phillips-Sul clubs:").pack(side=tk.LEFT)
        self.ps_chain = tk.BooleanVar(value=True)
        ttk.Checkbutton(ps, text="Chain growth index (prev. yr = 100)",
                        variable=self.ps_chain).pack(side=tk.LEFT, padx=6)
        self.ps_hp = tk.BooleanVar(value=True)
        ttk.Checkbutton(ps, text="HP trend (λ=400)",
                        variable=self.ps_hp).pack(side=tk.LEFT, padx=6)
        self.ps_merge = tk.BooleanVar(value=True)
        ttk.Checkbutton(ps, text="Merge clubs",
                        variable=self.ps_merge).pack(side=tk.LEFT, padx=6)
        ttk.Button(ps, text="log-t Test (all)",
                   command=self._run_logt).pack(side=tk.LEFT, padx=4)
        ttk.Button(ps, text="Club Clustering", style="Accent.TButton",
                   command=self._run_ps_clubs).pack(side=tk.LEFT, padx=4)
        ttk.Button(ps, text="Use Clubs as Clusters",
                   command=self._use_clubs_as_clusters).pack(
            side=tk.LEFT, padx=4)
        self.ps_result = None

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.conv_txt        = make_text(left, height=30, copy_root=self.root)
        self.conv_plot_frame = right
        self._refresh_conv_vars()

    def _refresh_conv_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.conv_var["values"] = num_cols
        self.conv_var.set("Y by L" if "Y by L" in num_cols
                          else (num_cols[0] if num_cols else ""))
        default_ctrl = {"Savings Percentage", "Human Capital Proxy",
                        "PIB towards research", "Labor in research"}
        self.conv_ctrl_lb.delete(0, tk.END)
        for c in num_cols:
            self.conv_ctrl_lb.insert(tk.END, c)
            if c in default_ctrl:
                self.conv_ctrl_lb.select_set(tk.END)

    def _get_conv_subset(self, include_controls=False):
        if not self._check_data():
            return None, None
        var = self.conv_var.get()
        if not var:
            self._refresh_conv_vars()
            var = self.conv_var.get()
        if not var:
            messagebox.showwarning("Config", "Select an income variable.")
            return None, None
        ctrl_cols = (
            [self.conv_ctrl_lb.get(i)
             for i in self.conv_ctrl_lb.curselection()]
            if include_controls else []
        )
        keep = ["Country", "Year", var] + [c for c in ctrl_cols if c != var]
        df   = self.df[keep].dropna()
        grp  = self.conv_group.get()
        if grp == "Western EU":
            df = df[df["Country"].str.lower().isin(WESTERN_EU)]
        elif grp == "Eastern EU":
            df = df[df["Country"].str.lower().isin(EASTERN_EU)]
        elif grp == "By Cluster" and self.clusters is not None:
            df = df.merge(self.clusters[["Country", "Cluster"]],
                          on="Country", how="left")
        return df, var

    # ── Absolute β-convergence ────────────────────────────────────────────────
    def _run_beta_conv(self):
        df, var = self._get_conv_subset()
        if df is None:
            return
        for w in self.conv_plot_frame.winfo_children():
            w.destroy()
        clear_txt(self.conv_txt)

        grp     = df.sort_values("Year").groupby("Country")[var]
        initial = grp.first()
        final   = grp.last()
        yrs     = (df.groupby("Country")["Year"]
                     .apply(lambda x: x.max() - x.min())
                     .replace(0, 1))
        growth  = ((np.log(final.clip(lower=1e-9)) -
                    np.log(initial.clip(lower=1e-9))) / yrs)
        valid   = pd.DataFrame({
            "log_y0": np.log(initial.clip(lower=1e-9)),
            "growth": growth,
        }).dropna()

        X   = sm.add_constant(valid["log_y0"])
        ols = sm.OLS(valid["growth"], X).fit()
        b   = ols.params["log_y0"]
        p   = ols.pvalues["log_y0"]
        T   = yrs.mean()
        lam = -np.log(1 + b * T) / T if (1 + b * T) > 0 else float("nan")
        hl  = np.log(2) / lam if lam and lam > 0 else float("nan")

        write(self.conv_txt,
              f"β-CONVERGENCE\n{'='*52}\n"
              f"Variable : {var}   Group: {self.conv_group.get()}\n"
              f"Countries: {len(valid)}\n\n"
              f"OLS: growth = α + β · ln(Y₀)\n"
              f"{'─'*52}\n"
              f"β coefficient  {b:>12.5f}\n"
              f"Std Error      {ols.bse['log_y0']:>12.5f}\n"
              f"t-statistic    {ols.tvalues['log_y0']:>12.3f}\n"
              f"p-value        {p:>12.4f}  {stars(p)}\n"
              f"R²             {ols.rsquared:>12.4f}\n"
              f"{'─'*52}\n"
              f"Speed (λ)      {lam:>12.5f}\n"
              f"Half-life      {hl:>12.1f} years\n\n"
              + ("→ ABSOLUTE β-CONVERGENCE confirmed (β<0, p<0.10)\n"
                 if b < 0 and p < 0.1
                 else "→ No significant β-convergence detected\n"))

        fig = Figure(figsize=(7, 5), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        ax.scatter(valid["log_y0"], valid["growth"],
                   color=theme.BLUE, alpha=0.8, s=60, zorder=3)
        xl = np.linspace(valid["log_y0"].min(), valid["log_y0"].max(), 100)
        ax.plot(xl, ols.params["const"] + b * xl,
                color=theme.TEAL, linewidth=2)
        ax.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        for nm, row in valid.iterrows():
            ax.annotate(nm, (row["log_y0"], row["growth"]),
                        fontsize=6, color=theme.FG, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("ln(Initial Income per Worker)", color=theme.FG)
        ax.set_ylabel("Avg Annual Growth Rate", color=theme.FG)
        ax.set_title(f"β-Convergence  (β={b:.4f}, p={p:.4f})", color=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.conv_plot_frame)
        self.status_var.set(f"β-convergence: β={b:.4f}  p={p:.4f}")

    # ── Conditional β-convergence ─────────────────────────────────────────────
    def _run_conditional_beta(self):
        df, var = self._get_conv_subset(include_controls=True)
        if df is None:
            return
        ctrl_cols = [c for c in df.columns
                     if c not in ("Country", "Year", var)]
        if not ctrl_cols:
            messagebox.showwarning("Conditional β",
                "Select at least one conditioning variable in the listbox.")
            return
        for w in self.conv_plot_frame.winfo_children():
            w.destroy()
        clear_txt(self.conv_txt)

        grp     = df.sort_values("Year").groupby("Country")[var]
        initial = grp.first(); final = grp.last()
        yrs     = (df.groupby("Country")["Year"]
                     .apply(lambda x: x.max() - x.min()).replace(0, 1))
        growth  = ((np.log(final.clip(lower=1e-9)) -
                    np.log(initial.clip(lower=1e-9))) / yrs)
        ctrl_avg = df.groupby("Country")[ctrl_cols].mean()
        valid    = pd.DataFrame({
            "log_y0": np.log(initial.clip(lower=1e-9)),
            "growth": growth,
        }).join(ctrl_avg).dropna()

        X   = sm.add_constant(valid[["log_y0"] + ctrl_cols])
        ols = sm.OLS(valid["growth"], X).fit()
        b   = ols.params["log_y0"]
        p   = ols.pvalues["log_y0"]
        T   = yrs.mean()
        lam = -np.log(1 + b * T) / T if (1 + b * T) > 0 else float("nan")
        hl  = np.log(2) / lam if lam and lam > 0 else float("nan")

        hdr  = (f"CONDITIONAL β-CONVERGENCE\n{'='*56}\n"
                f"Variable   : {var}   Group: {self.conv_group.get()}\n"
                f"Controls   : {', '.join(ctrl_cols)}\n"
                f"Countries  : {len(valid)}\n\n"
                f"OLS: growth = α + β·ln(Y₀) + Σγₖ·controls\n{'─'*56}\n")
        rows = (f"{'Variable':<30} {'Coef':>10} {'SE':>10} "
                f"{'p':>8} {'':>4}\n{'─'*56}\n")
        for v in ols.params.index:
            rows += (f"{v:<30} {ols.params[v]:>10.5f} {ols.bse[v]:>10.5f} "
                     f"{ols.pvalues[v]:>8.4f} {stars(ols.pvalues[v]):>4}\n")
        footer = (f"{'─'*56}\nR²={ols.rsquared:.4f}  Adj.R²={ols.rsquared_adj:.4f}\n\n"
                  f"β (conditional)  {b:>12.5f}\n"
                  f"Speed (λ)        {lam:>12.5f}\n"
                  f"Half-life        {hl:>12.1f} years\n\n"
                  + ("→ CONDITIONAL β-CONVERGENCE confirmed (β<0, p<0.10)\n"
                     if b < 0 and p < 0.1
                     else "→ No conditional convergence detected\n"))
        write(self.conv_txt, hdr + rows + footer)

        # Partial regression plot
        X_ctrl  = sm.add_constant(valid[ctrl_cols])
        res_g   = sm.OLS(valid["growth"],  X_ctrl).fit().resid
        res_y0  = sm.OLS(valid["log_y0"], X_ctrl).fit().resid
        slope_p, intercept_p, *_ = scipy_stats.linregress(res_y0, res_g)

        fig = Figure(figsize=(7, 5), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        ax.scatter(res_y0, res_g, color=theme.BLUE, alpha=0.8, s=60, zorder=3)
        xl = np.linspace(res_y0.min(), res_y0.max(), 100)
        ax.plot(xl, intercept_p + slope_p * xl,
                color=theme.TEAL, linewidth=2)
        ax.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        ax.axvline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        for nm, x_val, y_val in zip(valid.index, res_y0, res_g):
            ax.annotate(nm, (x_val, y_val), fontsize=6,
                        color=theme.FG, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("Residual ln(Y₀)  |  controls", color=theme.FG)
        ax.set_ylabel("Residual growth  |  controls", color=theme.FG)
        ax.set_title(
            f"Conditional β-Convergence (partial)  β={b:.4f}, p={p:.4f}",
            color=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.conv_plot_frame)
        self.status_var.set(
            f"Conditional β={b:.4f}  p={p:.4f}  R²={ols.rsquared:.4f}")

    # ── σ-convergence ─────────────────────────────────────────────────────────
    def _run_sigma_conv(self):
        df, var = self._get_conv_subset()
        if df is None:
            return
        for w in self.conv_plot_frame.winfo_children():
            w.destroy()
        clear_txt(self.conv_txt)

        sigma = (df.groupby("Year")[var]
                   .apply(lambda x: np.std(np.log(x.clip(lower=1e-9))))
                   .reset_index())
        sigma.columns = ["Year", "sigma"]
        x = sigma["Year"].values.astype(float)
        y = sigma["sigma"].values
        slope, intercept, r, pv, se = scipy_stats.linregress(x, y)

        write(self.conv_txt,
              f"σ-CONVERGENCE\n{'='*52}\n"
              f"Variable : {var}   Group: {self.conv_group.get()}\n"
              f"Measure  : Std Dev of ln({var}) across countries per year\n\n"
              f"Linear trend\n{'─'*52}\n"
              f"Slope       {slope:>12.5f} per year\n"
              f"Intercept   {intercept:>12.5f}\n"
              f"r           {r:>12.4f}\n"
              f"p-value     {pv:>12.4f}  {stars(pv)}\n\n"
              + ("→ σ-CONVERGENCE: dispersion is narrowing\n"
                 if slope < 0 and pv < 0.1
                 else ("→ σ-DIVERGENCE: dispersion is widening\n"
                       if slope > 0 and pv < 0.1
                       else "→ No significant trend in cross-country dispersion\n")))

        fig = Figure(figsize=(7, 5), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        ax.plot(sigma["Year"], sigma["sigma"],
                color=theme.BLUE, linewidth=2,
                marker="o", markersize=5, label="σ (Std Dev)")
        ax.plot(x, intercept + slope * x,
                color=theme.TEAL, linewidth=1.5, linestyle="--",
                label=f"Trend (slope={slope:.5f})")
        ax.fill_between(sigma["Year"], sigma["sigma"].min(), sigma["sigma"],
                        alpha=0.12, color=theme.BLUE)
        ax.set_xlabel("Year", color=theme.FG)
        ax.set_ylabel("σ = Std(ln Y/L)", color=theme.FG)
        ax.set_title("σ-Convergence: Cross-Country Dispersion over Time",
                     color=theme.FG)
        ax.legend(facecolor=theme.WBG, labelcolor=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.conv_plot_frame)
        self.status_var.set(f"σ-convergence: slope={slope:.5f}  p={pv:.4f}")

    # ── Both plots side by side ───────────────────────────────────────────────
    def _run_both_conv(self):
        df, var = self._get_conv_subset()
        if df is None:
            return
        for w in self.conv_plot_frame.winfo_children():
            w.destroy()

        grp     = df.sort_values("Year").groupby("Country")[var]
        initial = grp.first(); final = grp.last()
        yrs     = (df.groupby("Country")["Year"]
                     .apply(lambda x: x.max() - x.min()).replace(0, 1))
        growth  = ((np.log(final.clip(lower=1e-9)) -
                    np.log(initial.clip(lower=1e-9))) / yrs)
        valid   = pd.DataFrame({
            "log_y0": np.log(initial.clip(lower=1e-9)),
            "growth": growth,
        }).dropna()
        Xb  = sm.add_constant(valid["log_y0"])
        olb = sm.OLS(valid["growth"], Xb).fit()
        b   = olb.params["log_y0"]; pb = olb.pvalues["log_y0"]

        sigma = (df.groupby("Year")[var]
                   .apply(lambda x: np.std(np.log(x.clip(lower=1e-9)))))
        xs = sigma.index.values.astype(float)
        sl, ic, *_ = scipy_stats.linregress(xs, sigma.values)

        fig = Figure(figsize=(13, 5), facecolor=theme.BG)
        ax1 = fig.add_subplot(121)
        ax1.scatter(valid["log_y0"], valid["growth"],
                    color=theme.BLUE, alpha=0.8, s=50, zorder=3)
        xl = np.linspace(valid["log_y0"].min(), valid["log_y0"].max(), 100)
        ax1.plot(xl, olb.params["const"] + b * xl,
                 color=theme.TEAL, linewidth=2)
        for nm, row in valid.iterrows():
            ax1.annotate(nm, (row["log_y0"], row["growth"]),
                         fontsize=6, color=theme.FG, alpha=0.7,
                         xytext=(2, 2), textcoords="offset points")
        ax1.set_xlabel("ln(Initial Y/L)", color=theme.FG)
        ax1.set_ylabel("Avg Growth Rate", color=theme.FG)
        ax1.set_title(f"β-Convergence  (β={b:.4f}, p={pb:.4f})",
                      color=theme.FG)

        ax2 = fig.add_subplot(122)
        ax2.plot(sigma.index, sigma.values,
                 color=theme.BLUE, linewidth=2, marker="o", markersize=4)
        ax2.plot(xs, ic + sl * xs,
                 color=theme.TEAL, linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Year", color=theme.FG)
        ax2.set_ylabel("σ = Std(ln Y/L)", color=theme.FG)
        ax2.set_title(f"σ-Convergence  (slope={sl:.5f})", color=theme.FG)

        fig.suptitle(f"Convergence Analysis — {self.conv_group.get()}",
                     color=theme.FG, fontsize=13)
        fig.tight_layout()
        embed_figure(fig, self.conv_plot_frame)

    # ══════════════════════════════════════════════════════════════════════
    # Phillips-Sul (2007) log-t test and convergence clubs
    # ══════════════════════════════════════════════════════════════════════
    def _ps_panel(self):
        if not self._check_data():
            return None, None
        col = self.conv_var.get()
        if not col:
            messagebox.showwarning("Phillips-Sul", "Pick an income variable.")
            return None, None
        logx = clubs.prepare_panel(self.df, col, self.ps_chain.get(),
                                   400.0 if self.ps_hp.get() else None)
        if logx.shape[1] < 3 or logx.shape[0] < 10:
            messagebox.showwarning(
                "Phillips-Sul", "Need ≥ 3 countries with a complete, positive "
                "series and ≥ 10 years.")
            return None, None
        return col, logx

    def _ps_header(self, col, logx):
        how = ("chained growth index, base = 100" if self.ps_chain.get()
               else "levels as given (rescaled so log > 1)")
        return (f"  Variable : {col}  ({how})\n"
                f"  Filter   : {'HP trend, λ = 400' if self.ps_hp.get() else 'none'}"
                f"   |   N = {logx.shape[1]}, T = {logx.shape[0]}, "
                f"r = 0.3 (first {int(0.3 * logx.shape[0])} yrs dropped)\n")

    @staticmethod
    def _ps_line(lt):
        verdict = "converges" if lt.converges else "rejects convergence"
        return (f"b = {lt.b:+.3f}  (SE {lt.se:.3f})  t = {lt.t:+.2f}  "
                f"→ {verdict}")

    def _run_logt(self):
        col, logx = self._ps_panel()
        if logx is None:
            return
        lt = clubs.log_t(logx)
        clear_txt(self.conv_txt)
        write(self.conv_txt,
              f"PHILLIPS-SUL log-t CONVERGENCE TEST\n{'='*64}\n"
              + self._ps_header(col, logx) +
              f"\n  {self._ps_line(lt)}\n\n"
              "  H0: all countries converge (b ≥ 0). Reject if t < −1.65.\n"
              "  b ≥ 2 ⇒ convergence in levels; 0 ≤ b < 2 ⇒ conditional\n"
              "  convergence in growth rates. 2b ≈ speed of convergence.\n")
        self._plot_ps_paths(logx, None, col)

    def _run_ps_clubs(self):
        col, logx = self._ps_panel()
        if logx is None:
            return
        res = clubs.club_clustering(logx, merge=self.ps_merge.get())
        self.ps_result = res
        L = [f"PHILLIPS-SUL CLUB CONVERGENCE\n{'='*64}\n",
             self._ps_header(col, logx),
             f"\n  Full sample: {self._ps_line(res.full)}\n\n"]
        for k, (members, lt) in enumerate(zip(res.clubs, res.club_tests), 1):
            L.append(f"  CLUB {k}  ({len(members)} countries)   "
                     f"{self._ps_line(lt)}\n")
            L.append("    " + ", ".join(members) + "\n\n")
        if res.divergent:
            L.append("  NON-CONVERGENT: " + ", ".join(res.divergent) + "\n\n")
        if res.merges:
            L.append("  Merged (Schnurbus et al. 2017): "
                     + "; ".join(res.merges) + "\n")
            L.append(f"  Initial clubs before merging: "
                     f"{len(res.initial_clubs)}\n\n")
        L.append("  Clubs are ordered by the average of the last third of "
                 "the sample\n  (club 1 = highest). Core formation uses the "
                 "max-t rule; sieve c* = 0.\n")
        if self.ps_chain.get():
            L.append("\n  ⚠ Chained indices start every country at 100, so "
                     "clubs group\n  cumulative growth since the base year — "
                     "not absolute productivity\n  levels. Use PWT level data "
                     "(docs/DATA_SOURCES.md) for level clubs.\n")
        clear_txt(self.conv_txt)
        write(self.conv_txt, "".join(L))
        self._plot_ps_paths(logx, res, col)
        self.status_var.set(f"Phillips-Sul: {len(res.clubs)} club(s), "
                            f"{len(res.divergent)} non-convergent")

    def _plot_ps_paths(self, logx, res, col):
        for w in self.conv_plot_frame.winfo_children():
            w.destroy()
        h   = clubs.transition_paths(logx)
        # Distinct colours for any number of clubs: theme colours first,
        # then matplotlib's tab10 (skipping near-duplicates of grey).
        pal = list(dict.fromkeys(theme.cluster_palette()))
        pal += [c for c in ("#e377c2", "#bcbd22", "#17becf", "#8c564b",
                            "#ff7f0e", "#2ca02c", "#9467bd")]
        fig = Figure(figsize=(11, 7), facecolor=theme.BG)
        if res is None:
            ax = fig.add_subplot(111)
            for c in h.columns:
                ax.plot(h.index, h[c], lw=1, alpha=0.7)
            ax.axhline(1, color=theme.FG, lw=0.8, ls="--")
            ax.set_title(f"Relative transition paths h_it — {col}",
                         color=theme.FG)
        else:
            groups = [(f"Club {k}", m) for k, m in enumerate(res.clubs, 1)]
            if res.divergent:
                groups.append(("Non-convergent", res.divergent))
            n  = len(groups)
            gs = fig.add_gridspec(int(np.ceil(n / 2)), 3)
            ax0 = fig.add_subplot(gs[:, 0])
            for i, (name, m) in enumerate(groups):
                colr = pal[i % len(pal)]
                ax0.plot(h.index, h[m].mean(axis=1), color=colr, lw=2.2,
                         label=f"{name} (n={len(m)})")
            ax0.axhline(1, color=theme.FG, lw=0.8, ls="--")
            ax0.set_title("Average transition path by club", color=theme.FG)
            ax0.legend(fontsize=7)
            for i, (name, m) in enumerate(groups):
                ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
                for c in m:
                    ax.plot(h.index, h[c], color=pal[i % len(pal)], lw=0.9)
                ax.axhline(1, color=theme.FG, lw=0.5, ls="--")
                ax.set_title(name + ": " + ", ".join(m)[:48],
                             color=theme.FG, fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle(f"Phillips-Sul transition paths — {col}", color=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.conv_plot_frame)

    def _use_clubs_as_clusters(self):
        res = self.ps_result
        if res is None or not res.clubs:
            messagebox.showinfo("Clubs", "Run Club Clustering first.")
            return
        mem = res.membership()
        self.clusters = mem[mem["Club"] > 0].rename(
            columns={"Club": "Cluster"}).reset_index(drop=True)
        messagebox.showinfo(
            "Clubs",
            f"{len(res.clubs)} club(s) now used as clusters — Tab 2 "
            "(Groups → K-Means clusters) and Tab 8 cluster comparison.\n"
            + (f"Non-convergent countries excluded: "
               f"{', '.join(res.divergent)}" if res.divergent else ""))

