"""
tabs/tab_panel.py — Panel econometrics tab.

Thesis-aligned workflow
-----------------------
Implements the core panel specification of the thesis' augmented Solow
framework.  Default dependent variable is output per worker (Y by L);
default regressors include savings rate, human capital proxy, labour in
research, GERD intensity (PIB towards research) and patent activity.

Models supported
  • Pooled OLS
  • Fixed Effects (Entity)
  • Fixed Effects (Entity + Time)     — the "within" estimator
  • Random Effects
  • Arellano-Bond GMM                 — first-differenced IV (lags 2, 3)

Diagnostics
  • Hausman specification test (FE vs RE) — cluster-wise runs recommended.
  • Forest plot of statistically significant coefficients with 95% CIs.

Mixin: PanelTabMixin
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from linearmodels.iv import IV2SLS as IV2SLS_iv
from scipy import stats as scipy_stats
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import theme
from constants import stars
from helpers import make_text, write, clear_txt, embed_figure


class PanelTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — PANEL ECONOMETRICS
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_panel(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Model Configuration", padding=10)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Dependent variable:").pack(side=tk.LEFT)
        self.dep_var = ttk.Combobox(row0, width=22, state="readonly")
        self.dep_var.pack(side=tk.LEFT, padx=6)
        ttk.Button(row0, text="Refresh",
                   command=self._refresh_panel_vars).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl,
                  text="Regressors  (Ctrl+click for multiple):").pack(
            anchor=tk.W, pady=(6, 2))
        reg_frm = ttk.Frame(ctrl); reg_frm.pack(fill=tk.X, pady=2)
        sb_r = ttk.Scrollbar(reg_frm); sb_r.pack(side=tk.RIGHT, fill=tk.Y)
        self.reg_lb = tk.Listbox(
            reg_frm, selectmode=tk.MULTIPLE, height=5,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            font=("Segoe UI", 9), yscrollcommand=sb_r.set,
            exportselection=False,
        )
        self.reg_lb.pack(fill=tk.BOTH, expand=True)
        sb_r.config(command=self.reg_lb.yview)

        opts = ttk.Frame(ctrl); opts.pack(fill=tk.X, pady=4)
        ttk.Label(opts, text="Model:").pack(side=tk.LEFT)
        self.model_type = ttk.Combobox(opts, width=30, state="readonly", values=[
            "Pooled OLS",
            "Fixed Effects (Entity)",
            "Fixed Effects (Entity + Time)",
            "Random Effects",
            "Arellano-Bond GMM",
        ])
        self.model_type.current(2); self.model_type.pack(side=tk.LEFT, padx=6)
        self.log_xform = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Log-transform",
                        variable=self.log_xform).pack(side=tk.LEFT, padx=8)
        self.robust_se = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Clustered SE (entity)",
                        variable=self.robust_se).pack(side=tk.LEFT, padx=6)

        lag_row = ttk.Frame(ctrl); lag_row.pack(fill=tk.X, pady=3)
        ttk.Label(lag_row, text="Lag depth for regressors:").pack(side=tk.LEFT)
        self.panel_lag_depth = tk.IntVar(value=0)
        _spl = tk.Spinbox(lag_row, from_=0, to=5, width=3,
                          textvariable=self.panel_lag_depth,
                          bg=theme.WBG, fg=theme.FG, insertbackground=theme.FG)
        _spl.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_spl)
        ttk.Label(lag_row, text="years  (0 = none)",
                  foreground=theme.GRAY).pack(side=tk.LEFT, padx=4)
        ttk.Button(lag_row, text="Build Lag Variables",
                   command=self._build_panel_lags).pack(side=tk.LEFT, padx=8)
        ttk.Button(lag_row, text="Add R&D × HC Interactions",
                   command=self._add_interactions).pack(side=tk.LEFT, padx=4)

        btn_row = ttk.Frame(ctrl); btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Run Model", style="Accent.TButton",
                   command=self._run_panel).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Hausman Test (FE vs RE)",
                   command=self._run_hausman).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Plot Significant Variables",
                   command=self._plot_sig_vars).pack(side=tk.LEFT, padx=4)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=2)
        right = ttk.Frame(pane); pane.add(right, weight=3)
        self.panel_txt        = make_text(left, height=30, copy_root=self.root)
        self.panel_plot_frame = right
        self.panel_result     = None

        self._refresh_panel_vars()

    # ── Variable refresh ──────────────────────────────────────────────────────
    def _refresh_panel_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.dep_var["values"] = num_cols
        self.dep_var.set("Y by L" if "Y by L" in num_cols
                         else (num_cols[0] if num_cols else ""))
        self.reg_lb.delete(0, tk.END)
        default_regs = {"Savings Percentage", "Human Capital Proxy",
                        "Labor in research", "PIB towards research",
                        "Patents per capita", "Labor not in research"}
        for c in num_cols:
            self.reg_lb.insert(tk.END, c)
            if c in default_regs:
                self.reg_lb.select_set(tk.END)

    # ── Lag / interaction builders ────────────────────────────────────────────
    def _build_panel_lags(self):
        if not self._check_data():
            return
        depth = self.panel_lag_depth.get()
        if depth == 0:
            messagebox.showinfo("Lags", "Set lag depth ≥ 1 first.")
            return
        regs = [self.reg_lb.get(i) for i in self.reg_lb.curselection()]
        if not regs:
            messagebox.showwarning("Config", "Select regressors first.")
            return
        df    = self.df.sort_values(["Country", "Year"]).copy()
        added = []
        for col in regs:
            if col not in df.columns:
                continue
            for n in range(1, depth + 1):
                lag_name = f"{col}_lag{n}"
                df[lag_name] = df.groupby("Country")[col].shift(n)
                added.append(lag_name)
        self.df = df
        self._refresh_panel_vars()
        self.status_var.set(f"Built {len(added)} lag column(s) up to t−{depth}.")
        messagebox.showinfo("Lag Variables",
            f"Added {len(added)} lag column(s): {', '.join(added[:6])}"
            + (" …" if len(added) > 6 else ""))

    def _add_interactions(self):
        if not self._check_data():
            return
        rd_keywords = {"research", "gerd", "pib towards", "patents", "rnd", "r&d"}
        hc_keywords = {"human capital", "education", "schooling", "hc"}
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        rd_cols  = [c for c in num_cols
                    if any(k in c.lower() for k in rd_keywords)]
        hc_cols  = [c for c in num_cols
                    if any(k in c.lower() for k in hc_keywords)]
        if not rd_cols or not hc_cols:
            messagebox.showwarning("Interactions",
                "Could not auto-detect R&D and Human Capital columns.\n"
                "Ensure column names contain keywords like 'research', 'patents', "
                "'human capital', etc.")
            return
        df    = self.df.copy()
        added = []
        for r in rd_cols:
            for h in hc_cols:
                name = f"{r}×{h}"
                if name not in df.columns:
                    df[name] = df[r] * df[h]
                    added.append(name)
        self.df = df
        self._refresh_panel_vars()
        self.status_var.set(f"Added {len(added)} interaction term(s).")
        messagebox.showinfo("Interactions",
            f"Added {len(added)} term(s):\n" + "\n".join(added[:8])
            + ("\n…" if len(added) > 8 else ""))

    # ── Data prep ─────────────────────────────────────────────────────────────
    def _get_panel_data(self):
        dep  = self.dep_var.get()
        regs = [self.reg_lb.get(i) for i in self.reg_lb.curselection()]
        if not dep or not regs:
            messagebox.showwarning("Config",
                "Select a dependent variable and at least one regressor.")
            return None, None, None
        cols = ["Country", "Year", dep] + [r for r in regs if r != dep]
        df   = self.df[cols].dropna()
        if self.log_xform.get():
            for c in [dep] + regs:
                if c in df.columns:
                    col = df[c]
                    if (col > 0).all():
                        df = df.copy(); df[c] = np.log(col)
                    elif (col + 1 > 0).all():
                        df = df.copy(); df[c] = np.log1p(col.clip(lower=0))
        df = df.set_index(["Country", "Year"])
        df.index.names = ["entity", "time"]
        return df, dep, regs

    # ── Estimation ────────────────────────────────────────────────────────────
    def _run_panel(self):
        if not self._check_data():
            return
        df, dep, regs = self._get_panel_data()
        if df is None:
            return
        self.status_var.set("Running panel regression…")

        def run():
            try:
                endog     = df[dep]
                exog_vars = [r for r in regs if r in df.columns and r != dep]
                exog      = sm.add_constant(df[exog_vars])
                mtype     = self.model_type.get()
                clustered = self.robust_se.get()
                cov       = "clustered" if clustered else "unadjusted"
                cov_kw    = {"cluster_entity": True} if clustered else {}

                if mtype == "Pooled OLS":
                    res = PooledOLS(endog, exog).fit(cov_type=cov, **cov_kw)
                elif mtype == "Fixed Effects (Entity)":
                    res = PanelOLS(endog, exog, entity_effects=True).fit(
                        cov_type=cov, **cov_kw)
                elif mtype == "Fixed Effects (Entity + Time)":
                    res = PanelOLS(endog, exog, entity_effects=True,
                                   time_effects=True).fit(cov_type=cov, **cov_kw)
                elif mtype == "Arellano-Bond GMM":
                    all_vars = [dep] + exog_vars
                    df_lev   = df[all_vars]
                    df_fd    = df_lev.groupby(level="entity").diff()
                    df_l2    = df_lev.groupby(level="entity").shift(2)
                    df_l3    = df_lev.groupby(level="entity").shift(3)
                    instr    = pd.concat([
                        df_l2.rename(columns={c: f"{c}__z2" for c in all_vars}),
                        df_l3.rename(columns={c: f"{c}__z3" for c in all_vars}),
                    ], axis=1)
                    combined  = pd.concat([df_fd, instr], axis=1).dropna()
                    endog_fd  = combined[dep]
                    exog_fd   = sm.add_constant(combined[exog_vars])
                    instrs    = combined[
                        [f"{c}__z2" for c in exog_vars] +
                        [f"{c}__z3" for c in exog_vars]]
                    res = IV2SLS_iv(endog_fd, exog_fd, None, instrs).fit(
                        cov_type="robust")
                else:
                    res = RandomEffects(endog, exog).fit(cov_type=cov)

                self.panel_result = res
                self.root.after(0, lambda: self._display_panel(res, mtype))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("Regression Error", msg))

        threading.Thread(target=run, daemon=True).start()

    def _display_panel(self, res, mtype):
        clear_txt(self.panel_txt)
        self.panel_txt.configure(state=tk.NORMAL)
        self.panel_txt.tag_configure("hdr",  foreground=theme.BLUE)
        self.panel_txt.tag_configure("sig",  foreground=theme.TEAL)
        self.panel_txt.tag_configure("norm", foreground=theme.FG)

        def ins(msg, tag="norm"):
            self.panel_txt.insert(tk.END, msg, tag)

        ins(f"{'='*66}\n", "hdr")
        ins(f"  MODEL: {mtype}\n", "hdr")
        ins(f"{'='*66}\n", "hdr")
        r2_adj = getattr(res, "rsquared_adj",
                         getattr(res, "rsquared_between", float("nan")))
        ins(f"  R²: {res.rsquared:.4f}    Adj.R²: {r2_adj:.4f}"
            f"    N: {res.nobs:,}\n", "hdr")
        if hasattr(res, "f_statistic"):
            f = res.f_statistic
            ins(f"  F-stat: {f.stat:.3f}   p: {f.pval:.4f}\n", "hdr")
        ins(f"{'─'*66}\n", "hdr")
        ins(f"{'Variable':<28} {'Coef':>10} {'SE':>10} {'t':>8} "
            f"{'p-value':>9} {'Sig':>4}\n", "hdr")
        ins(f"{'─'*66}\n", "hdr")

        for var in res.params.index:
            coef = res.params[var]
            se   = res.std_errors[var]
            t    = res.tstats[var]
            p    = res.pvalues[var]
            s    = stars(p)
            line = (f"{var:<28} {coef:>10.4f} {se:>10.4f} "
                    f"{t:>8.3f} {p:>9.4f} {s:>4}\n")
            ins(line, "sig" if p < 0.1 else "norm")

        ins(f"{'─'*66}\n", "hdr")
        ins("  Significance: *** p<0.01  ** p<0.05  * p<0.1\n"
            "  Teal = statistically significant (p < 0.10)\n", "hdr")
        self.panel_txt.configure(state=tk.DISABLED)
        self.status_var.set(f"Panel OLS complete · R²={res.rsquared:.4f}")

    # ── Hausman test ──────────────────────────────────────────────────────────
    def _run_hausman(self):
        if not self._check_data():
            return
        df, dep, regs = self._get_panel_data()
        if df is None:
            return
        self.status_var.set("Running Hausman test…")

        def run():
            try:
                endog     = df[dep]
                exog_vars = [r for r in regs if r in df.columns and r != dep]
                exog      = sm.add_constant(df[exog_vars])
                fe_res    = PanelOLS(endog, exog, entity_effects=True).fit()
                re_res    = RandomEffects(endog, exog).fit()

                common = [c for c in
                          fe_res.params.index.intersection(re_res.params.index)
                          if c != "const"]
                if not common:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Hausman", "No common parameters to compare."))
                    return

                diff     = (fe_res.params[common].values -
                            re_res.params[common].values)
                cov_diff = (fe_res.cov.loc[common, common].values -
                            re_res.cov.loc[common, common].values)
                try:
                    stat = float(diff @ np.linalg.pinv(cov_diff) @ diff)
                except Exception:
                    stat = float(np.dot(diff, diff))
                dof = len(common)
                p   = float(1 - scipy_stats.chi2.cdf(stat, dof))

                msg = (f"HAUSMAN SPECIFICATION TEST (FE vs RE)\n{'='*50}\n"
                       f"H₀: Random Effects is consistent (prefer RE)\n"
                       f"H₁: FE is required (RE is inconsistent)\n\n"
                       f"χ²({dof}) = {stat:.4f}\n"
                       f"p-value  = {p:.4f}\n\n"
                       + ("→ REJECT H₀ at 5%  →  Use Fixed Effects\n"
                          if p < 0.05 else
                          "→ Fail to reject H₀  →  Random Effects acceptable\n"))
                stat_str = f"Hausman χ²={stat:.3f}  p={p:.4f}"
                self.root.after(0, lambda m=msg, s=stat_str: (
                    clear_txt(self.panel_txt),
                    write(self.panel_txt, m),
                    self.status_var.set(s),
                ))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("Hausman Error", msg))

        threading.Thread(target=run, daemon=True).start()

    # ── Forest plot ───────────────────────────────────────────────────────────
    def _plot_sig_vars(self):
        if self.panel_result is None:
            messagebox.showinfo("Info", "Run a model first.")
            return
        for w in self.panel_plot_frame.winfo_children():
            w.destroy()
        res   = self.panel_result
        pvals = res.pvalues
        sig   = [v for v in pvals.index if pvals[v] < 0.1]
        if not sig:
            messagebox.showinfo("Info",
                "No significant variables at the 10% level.")
            return

        # Sort by effect size so the forest plot tells a clean story
        order   = res.params[sig].abs().sort_values(ascending=True).index
        sig     = list(order)
        params  = res.params[sig]
        ci_low  = res.conf_int().loc[sig, "lower"]
        ci_high = res.conf_int().loc[sig, "upper"]
        colors  = [theme.TEAL if pvals[v] < 0.05 else theme.AMBER
                   for v in sig]

        fig = Figure(figsize=(8.5, max(4.2, len(sig) * 0.6 + 2)),
                     facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        y   = list(range(len(sig)))
        ax.barh(y, params.values, color=colors, alpha=0.88, zorder=3,
                height=0.6, edgecolor="none")
        ax.errorbar(params.values, y,
                    xerr=[params.values - ci_low.values,
                          ci_high.values - params.values],
                    fmt="none", color=theme.FG, capsize=4,
                    linewidth=1.4, zorder=4)
        # Label each bar with its value
        for yi, xv in zip(y, params.values):
            ha = "left" if xv >= 0 else "right"
            ax.text(xv + (0.02 if xv >= 0 else -0.02) * abs(params).max(),
                    yi, f"{xv:+.3f}", va="center", ha=ha,
                    fontsize=8.5, color=theme.FG, fontweight="medium")
        ax.axvline(0, color=theme.RED, linewidth=1.2, linestyle="--",
                   zorder=2, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sig, fontsize=9)
        ax.set_xlabel("Estimated coefficient (95% CI)")
        ax.set_title("Significant Coefficients · "
                     "Augmented Solow Panel Specification",
                     pad=10)
        ax.legend(
            handles=[Patch(color=theme.TEAL, label="p < 0.05  (significant)"),
                     Patch(color=theme.AMBER, label="p < 0.10  (marginal)")],
            loc="lower right",
        )
        fig.tight_layout()
        embed_figure(fig, self.panel_plot_frame)
