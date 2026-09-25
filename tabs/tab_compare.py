"""
tabs/tab_compare.py  —  Cross-cluster regression comparison with forest plot.
Mixin: CompareTabMixin
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import theme
from constants import stars
from helpers import make_text, write, clear_txt, embed_figure


class CompareTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 — CROSS-CLUSTER COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_compare(self, parent):
        ctrl = ttk.LabelFrame(
            parent,
            text="Cross-Cluster Regression Comparison  "
                 "(requires K=2 clustering)",
            padding=10,
        )
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Dependent variable:").pack(side=tk.LEFT)
        self.cmp_dep = ttk.Combobox(row0, width=22, state="readonly")
        self.cmp_dep.pack(side=tk.LEFT, padx=6)
        ttk.Button(row0, text="Refresh",
                   command=self._refresh_compare_vars).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="Regressors  (Ctrl+click):").pack(
            anchor=tk.W, pady=(6, 2))
        reg_frm2 = ttk.Frame(ctrl); reg_frm2.pack(fill=tk.X, pady=2)
        sb_r2 = ttk.Scrollbar(reg_frm2); sb_r2.pack(side=tk.RIGHT, fill=tk.Y)
        self.cmp_reg_lb = tk.Listbox(
            reg_frm2, selectmode=tk.MULTIPLE, height=4,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            font=("Segoe UI", 9), yscrollcommand=sb_r2.set,
            exportselection=False,
        )
        self.cmp_reg_lb.pack(fill=tk.BOTH, expand=True)
        sb_r2.config(command=self.cmp_reg_lb.yview)

        row1 = ttk.Frame(ctrl); row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Model:").pack(side=tk.LEFT)
        self.cmp_model = ttk.Combobox(row1, width=28, state="readonly",
                                      values=[
            "Fixed Effects (Entity)",
            "Fixed Effects (Entity + Time)",
            "Pooled OLS",
            "Random Effects",
        ])
        self.cmp_model.current(0); self.cmp_model.pack(side=tk.LEFT, padx=6)
        self.cmp_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Log-transform",
                        variable=self.cmp_log).pack(side=tk.LEFT, padx=8)

        ttk.Button(ctrl, text="Run Cluster Comparison",
                   style="Accent.TButton",
                   command=self._run_cluster_compare).pack(pady=6, anchor=tk.W)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.cmp_txt        = make_text(left, height=30, copy_root=self.root)
        self.cmp_plot_frame = right

    def _refresh_compare_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.cmp_dep["values"] = num_cols
        self.cmp_dep.set("Y by L" if "Y by L" in num_cols
                         else (num_cols[0] if num_cols else ""))
        default_regs = {"Savings Percentage", "Human Capital Proxy",
                        "Labor in research", "PIB towards research",
                        "Patents per capita"}
        self.cmp_reg_lb.delete(0, tk.END)
        for c in num_cols:
            self.cmp_reg_lb.insert(tk.END, c)
            if c in default_regs:
                self.cmp_reg_lb.select_set(tk.END)

    # ── Cluster comparison run ────────────────────────────────────────────────
    def _run_cluster_compare(self):
        if not self._check_data():
            return
        if self.clusters is None:
            messagebox.showwarning("No Clusters",
                "Run K=2 clustering first in the Clustering tab.")
            return
        n_clusters = self.clusters["Cluster"].nunique()
        if n_clusters != 2:
            messagebox.showwarning(
                "K≠2",
                f"Comparison requires exactly 2 clusters; found {n_clusters}.")
            return

        dep  = self.cmp_dep.get()
        regs = [self.cmp_reg_lb.get(i)
                for i in self.cmp_reg_lb.curselection()]
        if not dep or not regs:
            messagebox.showwarning("Config",
                "Select a dependent variable and at least one regressor.")
            return

        self.status_var.set("Running cross-cluster comparison…")

        def run():
            try:
                results = {}
                cl_ids  = sorted(self.clusters["Cluster"].unique())
                rd_kw   = {"research", "gerd", "pib towards",
                           "patents", "rnd", "r&d"}
                rd_cols = [c for c in regs
                           if any(k in c.lower() for k in rd_kw)] or regs
                cluster_means = {
                    cl: (self.df.merge(self.clusters, on="Country")
                                .query(f"Cluster == {cl}")[rd_cols]
                                .mean().mean())
                    for cl in cl_ids
                }
                inn_id   = max(cluster_means, key=cluster_means.get)
                cl_names = {cl: ("Innovative" if cl == inn_id else "Emerging")
                            for cl in cl_ids}

                for cl_id in cl_ids:
                    cl_name   = cl_names[cl_id]
                    countries = self.clusters[
                        self.clusters["Cluster"] == cl_id
                    ]["Country"].tolist()
                    df_sub = self.df[
                        self.df["Country"].isin(countries)].copy()
                    cols   = (["Country", "Year", dep] +
                              [r for r in regs if r != dep])
                    df_sub = df_sub[cols].dropna()
                    if self.cmp_log.get():
                        for c in [dep] + regs:
                            if c in df_sub.columns:
                                col = df_sub[c]
                                if (col > 0).all():
                                    df_sub[c] = np.log(col)
                    if len(df_sub) < 10:
                        results[cl_name] = None
                        continue
                    df_idx = df_sub.set_index(["Country", "Year"])
                    df_idx.index.names = ["entity", "time"]
                    endog  = df_idx[dep]
                    exog   = sm.add_constant(
                        df_idx[[r for r in regs
                                if r in df_idx.columns and r != dep]])
                    mtype  = self.cmp_model.get()
                    if mtype == "Pooled OLS":
                        res = PooledOLS(endog, exog).fit(
                            cov_type="clustered", cluster_entity=True)
                    elif mtype == "Fixed Effects (Entity)":
                        res = PanelOLS(endog, exog,
                                       entity_effects=True).fit(
                            cov_type="clustered", cluster_entity=True)
                    elif mtype == "Fixed Effects (Entity + Time)":
                        res = PanelOLS(endog, exog,
                                       entity_effects=True,
                                       time_effects=True).fit(
                            cov_type="clustered", cluster_entity=True)
                    else:
                        res = RandomEffects(endog, exog).fit(
                            cov_type="unadjusted")
                    results[cl_name] = res

                self.root.after(
                    0, lambda: self._display_compare(results, dep, regs))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("Compare Error", msg))

        threading.Thread(target=run, daemon=True).start()

    def _display_compare(self, results, dep, regs):
        clear_txt(self.cmp_txt)
        for w in self.cmp_plot_frame.winfo_children():
            w.destroy()

        for cl_name, res in results.items():
            if res is None:
                write(self.cmp_txt,
                      f"\n{cl_name}: insufficient observations (<10).\n")
                continue
            r2_adj = getattr(res, "rsquared_adj",
                             getattr(res, "rsquared_between", float("nan")))
            write(self.cmp_txt,
                  f"\n{'='*60}\n  {cl_name.upper()}  |  "
                  f"R²={res.rsquared:.4f}  Adj.R²={r2_adj:.4f}"
                  f"  N={res.nobs:,}\n{'─'*60}\n"
                  f"{'Variable':<28} {'Coef':>10} {'SE':>10} "
                  f"{'p':>8} {'':>4}\n{'─'*60}\n")
            for v in res.params.index:
                p = res.pvalues[v]
                write(self.cmp_txt,
                      f"{v:<28} {res.params[v]:>10.4f} "
                      f"{res.std_errors[v]:>10.4f} "
                      f"{p:>8.4f} {stars(p):>4}\n")

        # Forest plot — side-by-side 95% CI bars
        valid     = {k: v for k, v in results.items() if v is not None}
        if len(valid) < 2:
            self.status_var.set(
                "Comparison done (not enough clusters to plot)")
            return

        cl_names    = list(valid.keys())
        common_vars = [
            v for v in valid[cl_names[0]].params.index
            if v != "const" and all(v in valid[n].params.index
                                    for n in cl_names)
        ]
        if not common_vars:
            self.status_var.set("No common variables to plot.")
            return

        n_vars = len(common_vars)
        fig    = Figure(figsize=(11, max(4, n_vars * 0.7 + 2)),
                        facecolor=theme.BG)
        ax     = fig.add_subplot(111)
        pal    = [theme.TEAL, theme.BLUE]
        offset = 0.22
        y_pos  = np.arange(n_vars)

        for i, (cl_name, res) in enumerate(valid.items()):
            ci    = res.conf_int()
            coefs = res.params[common_vars].values
            lo    = ci.loc[common_vars, "lower"].values
            hi    = ci.loc[common_vars, "upper"].values
            y     = y_pos + (i - 0.5) * offset * 2
            ax.barh(y, coefs, height=offset * 1.6,
                    color=pal[i], alpha=0.75, label=cl_name, zorder=3)
            ax.errorbar(coefs, y,
                        xerr=[coefs - lo, hi - coefs],
                        fmt="none", color=theme.FG,
                        capsize=3, linewidth=1.2, zorder=4)

        ax.axvline(0, color=theme.RED, linewidth=1.2, linestyle="--", zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(common_vars, fontsize=9, color=theme.FG)
        ax.set_xlabel("Coefficient (with 95% CI)", color=theme.FG)
        ax.set_title(
            f"Coefficient Forest Plot — {cl_names[0]} vs {cl_names[1]}",
            color=theme.FG, pad=10)
        ax.legend(
            handles=[Patch(color=pal[i], label=n)
                     for i, n in enumerate(cl_names)],
            facecolor=theme.WBG, labelcolor=theme.FG,
        )
        fig.tight_layout()
        embed_figure(fig, self.cmp_plot_frame)
        self.status_var.set("Cross-cluster comparison complete.")
