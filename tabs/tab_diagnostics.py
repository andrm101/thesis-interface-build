"""
tabs/tab_diagnostics.py  —  Regression Diagnostics (Iteration 1).

OLS Residual Tests
  • Breusch-Pagan heteroskedasticity (chi² + F)
  • White's general heteroskedasticity test
  • Breusch-Godfrey serial correlation LM test (lags 1 & 2)
  • Jarque-Bera normality (skewness + kurtosis)
  • Ramsey RESET functional-form test
  • Variance Inflation Factors (VIF)
  • Durbin-Watson statistic

Panel-Specific Tests
  • Pesaran cross-sectional dependence (CD) test
  • Wooldridge first-difference AR(1) test
  • Modified Wald groupwise heteroskedasticity test

Residual Plots
  • Residuals vs Fitted
  • Scale-Location (√|e| vs Fitted)
  • Normal Q-Q plot
  • Residual histogram with normal overlay
  • CUSUM stability plot

Mixin: DiagnosticsTabMixin
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    acorr_breusch_godfrey,
    linear_reset,
)
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as scipy_stats
from matplotlib.figure import Figure

import theme
from constants import stars
from helpers import make_text, write, clear_txt, embed_figure


# ── Internal helpers ───────────────────────────────────────────────────────
def _pesaran_cd(resid_series: pd.Series):
    """
    Pesaran (2004) cross-sectional dependence test.
    H0: errors are cross-sectionally independent.
    Returns (CD_statistic, p_value).
    """
    resid_wide = resid_series.unstack(level=0)   # rows=time, cols=entity
    cols = resid_wide.columns.tolist()
    N = len(cols)
    if N < 2:
        return float("nan"), float("nan")
    rho_sum = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ci = resid_wide[cols[i]].dropna()
            cj = resid_wide[cols[j]].dropna()
            common_idx = ci.index.intersection(cj.index)
            T_ij = len(common_idx)
            if T_ij < 3:
                continue
            rho_ij = ci[common_idx].corr(cj[common_idx])
            if not np.isnan(rho_ij):
                rho_sum += np.sqrt(T_ij) * rho_ij
    scale = np.sqrt(2.0 / (N * (N - 1)))
    CD    = scale * rho_sum
    p_val = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(CD)))
    return CD, p_val


def _wooldridge_ar1(resid_series: pd.Series):
    """
    Wooldridge (2002) test for first-order serial correlation in panel data.
    Regress Δê_it on ê_i,t-1; under H0 (no autocorrelation) the coefficient ≈ -0.5.
    Returns (F_statistic, p_value).
    """
    resid_lag  = resid_series.groupby(level="entity").shift(1)
    resid_diff = resid_series.groupby(level="entity").diff()
    valid = pd.concat(
        {"lag": resid_lag, "diff": resid_diff}, axis=1
    ).dropna()
    if len(valid) < 10:
        return float("nan"), float("nan")
    ols_w = sm.OLS(valid["diff"],
                   sm.add_constant(valid["lag"])).fit()
    rho   = ols_w.params["lag"]
    se    = ols_w.bse["lag"]
    # F = ((rho - (-0.5)) / se)^2 / 1  ~  F(1, df_resid)
    F_stat = ((rho - (-0.5)) / se) ** 2
    p_val  = 1.0 - scipy_stats.f.cdf(F_stat, 1, ols_w.df_resid)
    return F_stat, p_val


def _modified_wald(resid_series: pd.Series):
    """
    Modified Wald test for groupwise heteroskedasticity.
    H0: σ²_i = σ² for all i.  W ~ chi²(N-1).
    """
    entity_var = resid_series.groupby(level="entity").var()
    entity_n   = resid_series.groupby(level="entity").count()
    N          = len(entity_var)
    if N < 2:
        return float("nan"), float("nan")
    # Greene (2012) approximation
    sigma2_bar = (entity_var * (entity_n - 1)).sum() / (entity_n - 1).sum()
    W = ((entity_n - 1) * ((entity_var / sigma2_bar) - 1) ** 2).sum()
    p_val = 1.0 - scipy_stats.chi2.cdf(W, N - 1)
    return W, p_val


class DiagnosticsTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9 — DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_diagnostics(self, parent):
        # ── Variable selection ─────────────────────────────────────────────
        vsel = ttk.LabelFrame(parent, text="Variable Selection", padding=10)
        vsel.pack(fill=tk.X, padx=12, pady=(8, 4))

        row0 = ttk.Frame(vsel); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Dependent variable:").pack(side=tk.LEFT)
        self.diag_dep = ttk.Combobox(row0, width=22, state="readonly")
        self.diag_dep.pack(side=tk.LEFT, padx=6)
        ttk.Button(row0, text="Refresh",
                   command=self._refresh_diag_vars).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text="← Sync from Panel Tab",
                   command=self._sync_diag_from_panel).pack(side=tk.LEFT, padx=4)

        ttk.Label(vsel,
                  text="Regressors  (Ctrl+click):").pack(anchor=tk.W, pady=(6, 2))
        reg_frm = ttk.Frame(vsel); reg_frm.pack(fill=tk.X, pady=2)
        sb_r = ttk.Scrollbar(reg_frm); sb_r.pack(side=tk.RIGHT, fill=tk.Y)
        self.diag_reg_lb = tk.Listbox(
            reg_frm, selectmode=tk.MULTIPLE, height=4,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            font=("Segoe UI", 9), yscrollcommand=sb_r.set,
            exportselection=False,
        )
        self.diag_reg_lb.pack(fill=tk.BOTH, expand=True)
        sb_r.config(command=self.diag_reg_lb.yview)

        opts = ttk.Frame(vsel); opts.pack(fill=tk.X, pady=3)
        self.diag_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Log-transform variables",
                        variable=self.diag_log).pack(side=tk.LEFT)

        # ── Action buttons ─────────────────────────────────────────────────
        btns = ttk.LabelFrame(parent, text="Run Diagnostics", padding=8)
        btns.pack(fill=tk.X, padx=12, pady=4)

        row1 = ttk.Frame(btns); row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Run All",
                   style="Accent.TButton",
                   command=self._run_all_diag).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="OLS Residual Tests",
                   command=self._run_ols_tests).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Panel Tests",
                   command=self._run_panel_tests).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(btns); row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="Residual Plots (2×3)",
                   command=self._plot_residuals).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="CUSUM Stability",
                   command=self._plot_cusum).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Partial Regression Plots",
                   command=self._plot_partial).pack(side=tk.LEFT, padx=4)

        _note = ttk.Label(
            btns,
            text="OLS Tests run pooled OLS (no FE).  "
                 "Panel Tests require a prior Panel-OLS run.",
            foreground=theme.AMBER,
        )
        _note.pack(anchor=tk.W, pady=(4, 0))
        self._special_labels.append((_note, "AMBER"))

        # ── Output pane ────────────────────────────────────────────────────
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=2)
        right = ttk.Frame(pane); pane.add(right, weight=3)
        self.diag_txt        = make_text(left, height=30,
                                         copy_root=self.root)
        self.diag_plot_frame = right

    # ── Variable helpers ───────────────────────────────────────────────────
    def _refresh_diag_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.diag_dep["values"] = num_cols
        if not self.diag_dep.get() or self.diag_dep.get() not in num_cols:
            self.diag_dep.set("Y by L" if "Y by L" in num_cols
                              else (num_cols[0] if num_cols else ""))
        self.diag_reg_lb.delete(0, tk.END)
        default = {"Savings Percentage", "Human Capital Proxy",
                   "Labor in research", "PIB towards research",
                   "Patents per capita"}
        for c in num_cols:
            self.diag_reg_lb.insert(tk.END, c)
            if c in default:
                self.diag_reg_lb.select_set(tk.END)

    def _sync_diag_from_panel(self):
        """Copy dep var + regressors selection from the Panel OLS tab."""
        if not hasattr(self, "dep_var") or self.dep_var.get() == "":
            messagebox.showinfo("Sync",
                "Run a Panel OLS model first, then sync.")
            return
        self._refresh_diag_vars()
        # Set dep var
        dep = self.dep_var.get()
        if dep in self.diag_dep["values"]:
            self.diag_dep.set(dep)
        # Mirror regressor selection
        sel_regs = set(self.reg_lb.get(i)
                       for i in self.reg_lb.curselection())
        self.diag_reg_lb.selection_clear(0, tk.END)
        for i in range(self.diag_reg_lb.size()):
            if self.diag_reg_lb.get(i) in sel_regs:
                self.diag_reg_lb.select_set(i)
        self.status_var.set("Diagnostics: synced from Panel OLS tab.")

    # ── Data prep (shared by test runners) ────────────────────────────────
    def _get_diag_ols(self):
        """Return (ols_result, exog_df, dep_name, reg_names) for pooled OLS,
        or None on error."""
        if not self._check_data():
            return None
        dep  = self.diag_dep.get()
        regs = [self.diag_reg_lb.get(i)
                for i in self.diag_reg_lb.curselection()]
        if not dep or not regs:
            messagebox.showwarning("Config",
                "Select a dependent variable and regressors.")
            return None
        cols = ["Country", "Year", dep] + [r for r in regs if r != dep]
        df   = self.df[cols].dropna()
        if self.diag_log.get():
            for c in [dep] + regs:
                if c in df.columns:
                    col = df[c]
                    if (col > 0).all():
                        df = df.copy(); df[c] = np.log(col)
                    elif (col + 1 > 0).all():
                        df = df.copy(); df[c] = np.log1p(col.clip(lower=0))
        y    = df[dep]
        X    = sm.add_constant(df[[r for r in regs if r in df.columns
                                   and r != dep]])
        res  = sm.OLS(y, X).fit()
        return res, X, dep, regs

    def _get_panel_resids(self):
        """Return residuals from the last panel run as a MultiIndex Series,
        or None if unavailable."""
        if self.panel_result is None:
            messagebox.showwarning("Panel Tests",
                "Run a model in the Panel OLS tab first.")
            return None
        try:
            resids = self.panel_result.resids
            resids.index.names = ["entity", "time"]
            return resids
        except Exception as e:
            messagebox.showerror("Panel Residuals", str(e))
            return None

    # ── OLS residual tests ─────────────────────────────────────────────────
    def _run_ols_tests(self):
        if not self._check_data():
            return
        result = self._get_diag_ols()
        if result is None:
            return
        res, X, dep, regs = result
        self.status_var.set("Running OLS residual tests…")

        def run():
            try:
                out = self._compute_ols_tests(res, X, dep)
                self.root.after(0, lambda: (
                    clear_txt(self.diag_txt),
                    write(self.diag_txt, out),
                    self.status_var.set("OLS diagnostics complete."),
                ))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("OLS Tests", msg))

        threading.Thread(target=run, daemon=True).start()

    def _compute_ols_tests(self, res, X, dep: str) -> str:
        resid   = res.resid
        fitted  = res.fittedvalues
        exog    = X.values
        # Names excluding constant
        reg_names = [c for c in X.columns if c != "const"]
        exog_no_c = X[[c for c in X.columns if c != "const"]].values

        lines = [
            f"OLS RESIDUAL DIAGNOSTICS\n{'='*62}\n",
            f"Dep. variable : {dep}\n",
            f"Regressors    : {', '.join(reg_names)}\n",
            f"Observations  : {int(res.nobs)}\n\n",
        ]

        # ── Durbin-Watson ──────────────────────────────────────────────────
        dw = durbin_watson(resid)
        lines.append(f"Durbin-Watson statistic : {dw:.4f}\n"
                     f"  (2.0 = no autocorrelation; <1 or >3 = concern)\n\n")

        # ── Breusch-Pagan ──────────────────────────────────────────────────
        try:
            bp_lm, bp_lmp, bp_f, bp_fp = het_breuschpagan(resid, exog_no_c)
            lines.append(
                f"HETEROSKEDASTICITY\n{'─'*62}\n"
                f"Breusch-Pagan (LM)  χ²={bp_lm:.4f}  p={bp_lmp:.4f}  "
                f"{stars(bp_lmp)}\n"
                f"  (F-form) F={bp_f:.4f}  p={bp_fp:.4f}\n"
                f"  {'→ Heteroskedasticity present' if bp_lmp < 0.05 else '→ No evidence of heteroskedasticity'}\n\n")
        except Exception as e:
            lines.append(f"Breusch-Pagan: error — {e}\n\n")

        # ── White's test ───────────────────────────────────────────────────
        try:
            wh_lm, wh_lmp, wh_f, wh_fp = het_white(resid, exog)
            lines.append(
                f"White's General Test  χ²={wh_lm:.4f}  p={wh_lmp:.4f}  "
                f"{stars(wh_lmp)}\n"
                f"  (F-form) F={wh_f:.4f}  p={wh_fp:.4f}\n"
                f"  {'→ Heteroskedasticity present' if wh_lmp < 0.05 else '→ Homoskedastic'}\n\n")
        except Exception as e:
            lines.append(f"White's test: error — {e}\n\n")

        # ── Breusch-Godfrey ────────────────────────────────────────────────
        try:
            bg1_lm, bg1_p, _, _ = acorr_breusch_godfrey(res, nlags=1)
            bg2_lm, bg2_p, _, _ = acorr_breusch_godfrey(res, nlags=2)
            lines.append(
                f"SERIAL CORRELATION\n{'─'*62}\n"
                f"Breusch-Godfrey  lag=1  LM={bg1_lm:.4f}  p={bg1_p:.4f}  "
                f"{stars(bg1_p)}\n"
                f"Breusch-Godfrey  lag=2  LM={bg2_lm:.4f}  p={bg2_p:.4f}  "
                f"{stars(bg2_p)}\n"
                f"  {'→ Serial correlation detected (lag 1)' if bg1_p < 0.05 else '→ No serial correlation (lag 1)'}\n\n")
        except Exception as e:
            lines.append(f"Breusch-Godfrey: error — {e}\n\n")

        # ── Jarque-Bera ────────────────────────────────────────────────────
        try:
            jb, jb_p, skew, kurt = jarque_bera(resid)
            lines.append(
                f"NORMALITY OF RESIDUALS\n{'─'*62}\n"
                f"Jarque-Bera  JB={jb:.4f}  p={jb_p:.4f}  {stars(jb_p)}\n"
                f"  Skewness={skew:.4f}  Excess Kurtosis={kurt:.4f}\n"
                f"  {'→ Non-normal residuals' if jb_p < 0.05 else '→ Residuals appear normal'}\n\n")
        except Exception as e:
            lines.append(f"Jarque-Bera: error — {e}\n\n")

        # ── Ramsey RESET ───────────────────────────────────────────────────
        try:
            reset = linear_reset(res, power=3, use_f=True)
            lines.append(
                f"FUNCTIONAL FORM\n{'─'*62}\n"
                f"Ramsey RESET (power=3)  F={reset.fvalue:.4f}  "
                f"p={reset.pvalue:.4f}  {stars(reset.pvalue)}\n"
                f"  {'→ Possible misspecification' if reset.pvalue < 0.05 else '→ No misspecification detected'}\n\n")
        except Exception as e:
            lines.append(f"Ramsey RESET: error — {e}\n\n")

        # ── VIF ────────────────────────────────────────────────────────────
        try:
            lines.append(f"MULTICOLLINEARITY (VIF)\n{'─'*62}\n")
            lines.append(f"{'Variable':<30} {'VIF':>8}\n")
            lines.append(f"{'─'*38}\n")
            for i, name in enumerate(reg_names):
                # VIF computed on exog_no_c (no constant column)
                vif_val = variance_inflation_factor(exog_no_c, i)
                flag = "  ← HIGH" if vif_val > 10 else (
                       "  ← moderate" if vif_val > 5 else "")
                lines.append(f"{name:<30} {vif_val:>8.2f}{flag}\n")
            lines.append("  (VIF > 5: moderate concern; VIF > 10: severe)\n\n")
        except Exception as e:
            lines.append(f"VIF: error — {e}\n\n")

        lines.append("─"*62 + "\n")
        lines.append("Significance: *** p<0.01  ** p<0.05  * p<0.1\n")
        return "".join(lines)

    # ── Panel-specific tests ───────────────────────────────────────────────
    def _run_panel_tests(self):
        resids = self._get_panel_resids()
        if resids is None:
            return
        self.status_var.set("Running panel diagnostic tests…")

        def run():
            try:
                out = self._compute_panel_tests(resids)
                self.root.after(0, lambda: (
                    clear_txt(self.diag_txt),
                    write(self.diag_txt, out),
                    self.status_var.set("Panel diagnostics complete."),
                ))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("Panel Tests", msg))

        threading.Thread(target=run, daemon=True).start()

    def _compute_panel_tests(self, resids: pd.Series) -> str:
        N  = resids.index.get_level_values("entity").nunique()
        T  = resids.index.get_level_values("time").nunique()
        lines = [
            f"PANEL DIAGNOSTIC TESTS\n{'='*62}\n",
            f"Entities (N) : {N}\n",
            f"Time periods (T) : {T}\n\n",
        ]

        # ── Pesaran CD ─────────────────────────────────────────────────────
        try:
            CD, p_CD = _pesaran_cd(resids)
            lines.append(
                f"CROSS-SECTIONAL DEPENDENCE\n{'─'*62}\n"
                f"Pesaran CD test  CD={CD:.4f}  p={p_CD:.4f}  "
                f"{stars(p_CD)}\n"
                f"  H0: errors are cross-sectionally independent\n"
                f"  {'→ Cross-sectional dependence detected — use PCSE or Driscoll-Kraay SE' if p_CD < 0.05 else '→ No significant cross-sectional dependence'}\n\n")
        except Exception as e:
            lines.append(f"Pesaran CD: error — {e}\n\n")

        # ── Wooldridge AR(1) ───────────────────────────────────────────────
        try:
            F_w, p_w = _wooldridge_ar1(resids)
            lines.append(
                f"SERIAL CORRELATION (Panel)\n{'─'*62}\n"
                f"Wooldridge AR(1) test  F={F_w:.4f}  p={p_w:.4f}  "
                f"{stars(p_w)}\n"
                f"  H0: no first-order autocorrelation in panel errors\n"
                f"  {'→ AR(1) serial correlation present — consider AR residuals or lags' if p_w < 0.05 else '→ No evidence of AR(1) autocorrelation'}\n\n")
        except Exception as e:
            lines.append(f"Wooldridge test: error — {e}\n\n")

        # ── Modified Wald ──────────────────────────────────────────────────
        try:
            W_mw, p_mw = _modified_wald(resids)
            lines.append(
                f"GROUPWISE HETEROSKEDASTICITY\n{'─'*62}\n"
                f"Modified Wald  W={W_mw:.4f}  p={p_mw:.4f}  "
                f"{stars(p_mw)}\n"
                f"  H0: σ²_i = σ² for all entities  [chi²({N-1})]\n"
                f"  {'→ Groupwise heteroskedasticity — use entity-robust SE' if p_mw < 0.05 else '→ Variances appear homogeneous across entities'}\n\n")
        except Exception as e:
            lines.append(f"Modified Wald: error — {e}\n\n")

        lines.append("─"*62 + "\n")
        lines.append("Significance: *** p<0.01  ** p<0.05  * p<0.1\n")
        return "".join(lines)

    # ── Run All ────────────────────────────────────────────────────────────
    def _run_all_diag(self):
        if not self._check_data():
            return
        result = self._get_diag_ols()
        if result is None:
            return
        res, X, dep, regs = result
        panel_resids = (self.panel_result.resids
                        if self.panel_result is not None else None)
        if panel_resids is not None:
            try:
                panel_resids.index.names = ["entity", "time"]
            except Exception:
                panel_resids = None
        self.status_var.set("Running all diagnostics…")

        def run():
            try:
                ols_out   = self._compute_ols_tests(res, X, dep)
                panel_out = (self._compute_panel_tests(panel_resids)
                             if panel_resids is not None
                             else "\n[Panel tests skipped — run Panel OLS first]\n")
                full = ols_out + "\n" + panel_out
                self.root.after(0, lambda: (
                    clear_txt(self.diag_txt),
                    write(self.diag_txt, full),
                    self.status_var.set("All diagnostics complete."),
                ))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("Diagnostics", msg))

        threading.Thread(target=run, daemon=True).start()

    # ── Residual plots (2 × 3 grid) ────────────────────────────────────────
    def _plot_residuals(self):
        if not self._check_data():
            return
        result = self._get_diag_ols()
        if result is None:
            return
        res, X, dep, regs = result
        for w in self.diag_plot_frame.winfo_children():
            w.destroy()

        resid  = res.resid.values
        fitted = res.fittedvalues.values
        std_r  = (resid - resid.mean()) / resid.std()

        fig = Figure(figsize=(13, 8), facecolor=theme.BG)
        axes = [fig.add_subplot(2, 3, i) for i in range(1, 7)]

        # 1. Residuals vs Fitted
        ax = axes[0]
        ax.scatter(fitted, resid, color=theme.BLUE, alpha=0.5, s=18, zorder=3)
        ax.axhline(0, color=theme.RED, linewidth=1, linestyle="--")
        ax.set_xlabel("Fitted values", color=theme.FG, fontsize=8)
        ax.set_ylabel("Residuals",     color=theme.FG, fontsize=8)
        ax.set_title("Residuals vs Fitted", color=theme.FG, fontsize=9)

        # 2. Scale-Location
        ax = axes[1]
        ax.scatter(fitted, np.sqrt(np.abs(std_r)),
                   color=theme.TEAL, alpha=0.5, s=18, zorder=3)
        ax.axhline(np.sqrt(np.abs(std_r)).mean(),
                   color=theme.RED, linewidth=1, linestyle="--")
        ax.set_xlabel("Fitted values",       color=theme.FG, fontsize=8)
        ax.set_ylabel("√|Std Residuals|",    color=theme.FG, fontsize=8)
        ax.set_title("Scale-Location",       color=theme.FG, fontsize=9)

        # 3. Normal Q-Q
        ax = axes[2]
        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid,
                                                                   dist="norm")
        ax.scatter(osm, osr, color=theme.BLUE, alpha=0.6, s=18, zorder=3)
        xl = np.array([osm[0], osm[-1]])
        ax.plot(xl, slope * xl + intercept,
                color=theme.TEAL, linewidth=1.8)
        ax.set_xlabel("Theoretical Quantiles", color=theme.FG, fontsize=8)
        ax.set_ylabel("Sample Quantiles",       color=theme.FG, fontsize=8)
        ax.set_title("Normal Q-Q",              color=theme.FG, fontsize=9)

        # 4. Residual histogram
        ax = axes[3]
        ax.hist(resid, bins=25, color=theme.BLUE, edgecolor=theme.DARK,
                alpha=0.8, density=True)
        xn = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(xn,
                scipy_stats.norm.pdf(xn, resid.mean(), resid.std()),
                color=theme.TEAL, linewidth=2, label="N(μ,σ)")
        ax.set_xlabel("Residuals",             color=theme.FG, fontsize=8)
        ax.set_ylabel("Density",               color=theme.FG, fontsize=8)
        ax.set_title("Residual Distribution",  color=theme.FG, fontsize=9)
        ax.legend(facecolor=theme.WBG, labelcolor=theme.FG, fontsize=7)

        # 5. Residuals vs Order (time/index)
        ax = axes[4]
        ax.plot(range(len(resid)), resid,
                color=theme.BLUE, linewidth=0.8, alpha=0.8)
        ax.axhline(0, color=theme.RED, linewidth=1, linestyle="--")
        ax.set_xlabel("Observation order", color=theme.FG, fontsize=8)
        ax.set_ylabel("Residuals",         color=theme.FG, fontsize=8)
        ax.set_title("Residuals vs Order", color=theme.FG, fontsize=9)

        # 6. Cook's Distance (influence)
        ax = axes[5]
        try:
            from statsmodels.stats.outliers_influence import OLSInfluence
            inf   = OLSInfluence(res)
            cooks = inf.cooks_distance[0]
            ax.stem(range(len(cooks)), cooks,
                    linefmt=f"{theme.BLUE}80",
                    markerfmt=f"o",
                    basefmt=f"{theme.GRAY}80")
            threshold = 4 / len(cooks)
            ax.axhline(threshold, color=theme.RED,
                       linewidth=1, linestyle="--",
                       label=f"4/n = {threshold:.4f}")
            ax.set_xlabel("Observation",  color=theme.FG, fontsize=8)
            ax.set_ylabel("Cook's D",     color=theme.FG, fontsize=8)
            ax.set_title("Cook's Distance (Influence)",
                         color=theme.FG, fontsize=9)
            ax.legend(facecolor=theme.WBG, labelcolor=theme.FG, fontsize=7)
        except Exception:
            ax.set_visible(False)

        for ax in axes:
            ax.tick_params(colors=theme.FG, labelsize=7)

        fig.suptitle(f"Residual Diagnostics — {dep}", color=theme.FG,
                     fontsize=11)
        fig.tight_layout()
        embed_figure(fig, self.diag_plot_frame)
        self.status_var.set("Residual plots drawn.")

    # ── CUSUM stability plot ───────────────────────────────────────────────
    def _plot_cusum(self):
        if not self._check_data():
            return
        result = self._get_diag_ols()
        if result is None:
            return
        res, X, dep, regs = result
        for w in self.diag_plot_frame.winfo_children():
            w.destroy()

        resid  = res.resid.values
        n      = len(resid)
        std_r  = resid / (resid.std() + 1e-12)
        cusum  = np.cumsum(std_r)
        # 5% significance bounds (Ploberger & Krämer approximation)
        bound = 0.948 * np.sqrt(n)

        fig = Figure(figsize=(10, 5), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        ax.plot(range(n), cusum, color=theme.BLUE, linewidth=1.8,
                label="CUSUM")
        ax.axhline( bound, color=theme.RED, linewidth=1.2,
                   linestyle="--", label="5% bounds (±0.948√n)")
        ax.axhline(-bound, color=theme.RED, linewidth=1.2, linestyle="--")
        ax.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle=":")
        # Shade region between bounds
        ax.fill_between(range(n), -bound, bound,
                        alpha=0.06, color=theme.TEAL)
        ax.set_xlabel("Observation", color=theme.FG)
        ax.set_ylabel("Cumulative Sum of Standardised Residuals",
                      color=theme.FG)
        ax.set_title(f"CUSUM Stability Test — {dep}", color=theme.FG)
        ax.legend(facecolor=theme.WBG, labelcolor=theme.FG)
        ax.tick_params(colors=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.diag_plot_frame)
        self.status_var.set("CUSUM plot drawn.")

    # ── Partial regression plots ───────────────────────────────────────────
    def _plot_partial(self):
        if not self._check_data():
            return
        result = self._get_diag_ols()
        if result is None:
            return
        res, X, dep, regs = result
        reg_names = [c for c in X.columns if c != "const"]
        if not reg_names:
            return
        for w in self.diag_plot_frame.winfo_children():
            w.destroy()

        n    = len(reg_names)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig  = Figure(figsize=(12, 4 * rows), facecolor=theme.BG)
        y    = res.model.endog

        for idx, name in enumerate(reg_names, 1):
            ax        = fig.add_subplot(rows, cols, idx)
            others    = [c for c in X.columns if c != name]
            X_others  = X[others]
            # Residualise y and x on all other regressors
            res_y     = sm.OLS(y, X_others).fit().resid
            res_x     = sm.OLS(X[name], X_others).fit().resid
            slope, ic, r, pv, _ = scipy_stats.linregress(res_x, res_y)
            xl        = np.linspace(res_x.min(), res_x.max(), 100)
            ax.scatter(res_x, res_y, color=theme.BLUE, alpha=0.5,
                       s=18, zorder=3)
            ax.plot(xl, ic + slope * xl, color=theme.TEAL, linewidth=2)
            ax.axhline(0, color=theme.GRAY, linewidth=0.6, linestyle=":")
            ax.set_xlabel(f"e({name} | others)", color=theme.FG, fontsize=8)
            ax.set_ylabel(f"e({dep} | others)",  color=theme.FG, fontsize=8)
            ax.set_title(f"{name}  (slope={slope:.3f}, p={pv:.3f})",
                         color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=7)

        fig.suptitle("Partial Regression (Added-Variable) Plots",
                     color=theme.FG, fontsize=11)
        fig.tight_layout()
        embed_figure(fig, self.diag_plot_frame)
        self.status_var.set("Partial regression plots drawn.")

    # ── Tab-change refresh ─────────────────────────────────────────────────
    def _refresh_diag_vars_tab(self):
        self._refresh_diag_vars()
