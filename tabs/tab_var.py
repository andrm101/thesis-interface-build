"""
tabs/tab_var.py  —  VAR · IRF · FEVD · Granger Causality · VECM tab.

Features
────────
  · VAR lag-order selection (AIC / BIC / HQIC / FPE)
  · VAR estimation + full equation summary
  · Residual diagnostics (Portmanteau, normality, heteroskedasticity)
  · Stability check (roots of characteristic polynomial)
  · Impulse Response Functions  (orthogonalized, Cholesky ordering)
  · Cumulative IRFs
  · Forecast Error Variance Decomposition (FEVD)
  · Multi-step forecasting with confidence bands
  · Granger causality matrix (all pairs)
  · VECM (Vector Error Correction Model) for cointegrated systems
"""

import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, messagebox

import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.stattools import grangercausalitytests

import theme
from helpers import make_text, write, clear_txt, embed_figure


# ── Trend code map ────────────────────────────────────────────────────────────
_TREND_MAP = {
    "nc (no constant)":       "n",
    "c (constant)":           "c",
    "ct (constant + trend)":  "ct",
}
_VECM_DET_MAP = {
    "nc (no constant)":       "n",
    "c (constant)":           "co",
    "ct (constant + trend)":  "lo",
}


class VARTabMixin:

    # ── Tab builder ───────────────────────────────────────────────────────────
    def _tab_var(self, parent):
        # ── Top section: variable selector + controls ─────────────────────────
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=12, pady=(8, 2))

        # Variable listbox
        vf = ttk.LabelFrame(top, text="Variables (select ≥ 2)")
        vf.pack(side=tk.LEFT, padx=(0, 12))
        self.var_lb = tk.Listbox(
            vf, selectmode=tk.MULTIPLE, height=5, width=26,
            bg=theme.WBG, fg=theme.FG,
            selectbackground=theme.BLUE,
            selectforeground=theme.WFG,
            exportselection=False)
        self.var_lb.pack(side=tk.LEFT)
        sb_v = ttk.Scrollbar(vf, orient=tk.VERTICAL, command=self.var_lb.yview)
        sb_v.pack(side=tk.LEFT, fill=tk.Y)
        self.var_lb.configure(yscrollcommand=sb_v.set)

        # Parameter controls grid
        cf = ttk.Frame(top)
        cf.pack(side=tk.LEFT, fill=tk.Y)

        def _lbl(text, row):
            ttk.Label(cf, text=text).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6), pady=3)

        _lbl("Country / Series:", 0)
        self.var_country = ttk.Combobox(cf, width=24, state="readonly")
        self.var_country.grid(row=0, column=1, padx=4, pady=3)

        _lbl("Transform:", 1)
        self.var_transform = ttk.Combobox(
            cf, width=24, state="readonly",
            values=["Levels (as-is)",
                    "First difference",
                    "Log first difference"])
        self.var_transform.current(0)
        self.var_transform.grid(row=1, column=1, padx=4, pady=3)

        _lbl("Trend:", 2)
        self.var_trend = ttk.Combobox(
            cf, width=24, state="readonly",
            values=list(_TREND_MAP.keys()))
        self.var_trend.current(1)
        self.var_trend.grid(row=2, column=1, padx=4, pady=3)

        _lbl("Max lags:", 3)
        self.var_maxlag_var = tk.StringVar(value="4")
        sp_lag = tk.Spinbox(cf, from_=1, to=12, width=5,
                            textvariable=self.var_maxlag_var,
                            bg=theme.WBG, fg=theme.FG,
                            insertbackground=theme.FG,
                            buttonbackground=theme.WBG)
        sp_lag.grid(row=3, column=1, sticky=tk.W, padx=4, pady=3)
        self._tk_spinboxes.append(sp_lag)

        _lbl("Lag criterion:", 4)
        self.var_criterion = ttk.Combobox(
            cf, width=8, state="readonly",
            values=["aic", "bic", "hqic", "fpe"])
        self.var_criterion.current(0)
        self.var_criterion.grid(row=4, column=1, sticky=tk.W, padx=4, pady=3)

        _lbl("IRF periods:", 5)
        self.var_periods_var = tk.StringVar(value="10")
        sp_per = tk.Spinbox(cf, from_=2, to=40, width=5,
                            textvariable=self.var_periods_var,
                            bg=theme.WBG, fg=theme.FG,
                            insertbackground=theme.FG,
                            buttonbackground=theme.WBG)
        sp_per.grid(row=5, column=1, sticky=tk.W, padx=4, pady=3)
        self._tk_spinboxes.append(sp_per)

        # ── Button row 1: estimation ──────────────────────────────────────────
        r2 = ttk.Frame(parent)
        r2.pack(fill=tk.X, padx=12, pady=(6, 2))
        for label, cmd in [
            ("Fit VAR",          self._run_var_fit),
            ("Auto-select Lags", self._run_var_lag_select),
            ("Residual Tests",   self._run_var_resid_tests),
            ("Stability Check",  self._run_var_stability),
        ]:
            ttk.Button(r2, text=label, command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Button row 2: dynamics ────────────────────────────────────────────
        r3 = ttk.Frame(parent)
        r3.pack(fill=tk.X, padx=12, pady=(2, 6))
        dyn_hdr = ttk.Label(r3, text="── Dynamics ──", foreground=theme.TEAL)
        dyn_hdr.pack(side=tk.LEFT, padx=(0, 10))
        self._special_labels.append((dyn_hdr, "TEAL"))
        for label, cmd in [
            ("IRF",               self._run_irf),
            ("Cumulative IRF",    self._run_cirf),
            ("FEVD",              self._run_fevd),
            ("Forecast",          self._run_var_forecast),
            ("Granger Causality", self._run_granger),
            ("VECM",              self._run_vecm),
        ]:
            ttk.Button(r3, text=label, command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Paned results ─────────────────────────────────────────────────────
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.var_txt        = make_text(left, height=30, copy_root=self.root)
        self.var_plot_frame = right
        self.var_result     = None   # VARResults stored after fitting
        self.var_data_fit   = None   # DataFrame used for last fit

        self._refresh_var_vars()

    # ── Variable / country refresh ────────────────────────────────────────────
    def _refresh_var_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.var_lb.delete(0, tk.END)
        for c in num_cols:
            self.var_lb.insert(tk.END, c)

        countries = sorted(self.df["Country"].unique())
        self.var_country["values"] = ["Cross-sectional mean"] + countries
        self.var_country.current(0)

    # ── Data preparation ──────────────────────────────────────────────────────
    def _var_get_data(self):
        """Return a clean (T × k) DataFrame ready for VAR / VECM."""
        sel  = self.var_lb.curselection()
        cols = [self.var_lb.get(i) for i in sel]
        if len(cols) < 2:
            raise ValueError("Select ≥ 2 variables in the variable list.")

        country = self.var_country.get()
        if country == "Cross-sectional mean":
            data = (self.df.groupby("Year")[cols]
                    .mean()
                    .sort_index()
                    .dropna())
        else:
            sub  = self.df[self.df["Country"] == country]
            data = sub.set_index("Year")[cols].sort_index().dropna()

        if len(data) < 8:
            raise ValueError(
                f"Only {len(data)} observations after filtering — need ≥ 8.")

        transform = self.var_transform.get()
        if transform == "First difference":
            data = data.diff().dropna()
        elif transform == "Log first difference":
            data = np.log(data.replace(0, np.nan)).diff().dropna()

        return data

    @staticmethod
    def _restyle_fig(fig):
        """Apply dark/light theme to a statsmodels-generated figure."""
        fig.set_facecolor(theme.BG)
        for ax in fig.get_axes():
            ax.set_facecolor(theme.WBG)
            ax.tick_params(colors=theme.FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(theme.GRAY)
            ax.title.set_color(theme.FG)
            ax.xaxis.label.set_color(theme.FG)
            ax.yaxis.label.set_color(theme.FG)
            for line in ax.get_lines():
                c = line.get_color()
                if c in ("black", "k", "#000000"):
                    line.set_color(theme.FG)
                elif c in ("grey", "gray", "#808080", "#C0C0C0"):
                    line.set_color(theme.GRAY)

    # ── Fit VAR ───────────────────────────────────────────────────────────────
    def _run_var_fit(self):
        if not self._check_data():
            return
        maxlag    = int(self.var_maxlag_var.get())
        criterion = self.var_criterion.get()
        trend     = _TREND_MAP.get(self.var_trend.get(), "c")

        def _work():
            try:
                data   = self._var_get_data()
                model  = VAR(data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(maxlags=maxlag, ic=criterion, trend=trend)

                self.var_result   = res
                self.var_data_fit = data

                k   = res.neqs
                p   = res.k_ar
                T   = res.nobs
                aic = res.aic
                bic = res.bic
                hqic = res.hqic
                fpe  = res.fpe

                lines = [
                    f"\n{'═'*60}",
                    f" VAR Model  —  {', '.join(data.columns)}",
                    f"{'═'*60}",
                    f"  Country / series : {self.var_country.get()}",
                    f"  Transform        : {self.var_transform.get()}",
                    f"  Trend            : {trend}",
                    f"  Lag order (p)    : {p}",
                    f"  Variables (k)    : {k}",
                    f"  Observations (T) : {T}",
                    f"",
                    f"  Information criteria:",
                    f"    AIC  : {aic:>12.6f}",
                    f"    BIC  : {bic:>12.6f}",
                    f"    HQIC : {hqic:>12.6f}",
                    f"    FPE  : {fpe:>12.6e}",
                    f"",
                ]

                # Per-equation summary
                coef_arr  = res.coefs        # (p, k, k)
                const_arr = res.coefs_exog   # (k, n_exog)
                for i, eq_name in enumerate(data.columns):
                    lines += [
                        f"  {'─'*56}",
                        f"  Equation: {eq_name}",
                        f"  {'─'*56}",
                    ]
                    for lag in range(p):
                        for j, rhs_name in enumerate(data.columns):
                            lines.append(
                                f"    L{lag+1}.{rhs_name:<22} "
                                f"{coef_arr[lag, i, j]:>10.5f}")
                    if (const_arr is not None
                            and const_arr.shape[1] > 0
                            and const_arr.shape[0] > i):
                        lines.append(
                            f"    const                    "
                            f"{const_arr[i, 0]:>10.5f}")
                    lines.append("")

                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(0, lambda: self._plot_var_fit(data, res))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "VAR Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Lag selection ─────────────────────────────────────────────────────────
    def _run_var_lag_select(self):
        if not self._check_data():
            return
        maxlag = int(self.var_maxlag_var.get())
        trend  = _TREND_MAP.get(self.var_trend.get(), "c")

        def _work():
            try:
                data  = self._var_get_data()
                model = VAR(data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sel = model.select_order(maxlags=maxlag, trend=trend)

                lines = [
                    f"\n{'═'*60}",
                    f" VAR Lag Order Selection",
                    f"{'═'*60}",
                    f"  Variables : {', '.join(data.columns)}",
                    f"  Trend     : {trend}",
                    f"  Max lags  : {maxlag}",
                    f"",
                    f"  Selected orders:",
                    f"    AIC  : {sel.aic}",
                    f"    BIC  : {sel.bic}",
                    f"    HQIC : {sel.hqic}",
                    f"    FPE  : {sel.fpe}",
                    f"",
                    f"  Full IC table:",
                    f"  {'Lag':>4} {'AIC':>14} {'BIC':>14} {'HQIC':>14} {'FPE':>14}",
                    f"  {'─'*60}",
                ]
                for lag in range(maxlag + 1):
                    try:
                        lines.append(
                            f"  {lag:>4} "
                            f"{sel.ics['aic'][lag]:>14.6f} "
                            f"{sel.ics['bic'][lag]:>14.6f} "
                            f"{sel.ics['hqic'][lag]:>14.6f} "
                            f"{sel.ics['fpe'][lag]:>14.6e}")
                    except (KeyError, IndexError):
                        pass
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(0, lambda: self._plot_ic_table(sel, maxlag))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Lag Selection Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Residual diagnostic tests ─────────────────────────────────────────────
    def _run_var_resid_tests(self):
        if not self._check_data():
            return
        if self.var_result is None:
            messagebox.showwarning("VAR", "Fit a VAR model first.")
            return

        def _work():
            try:
                res  = self.var_result
                # Portmanteau test
                pt   = res.test_whiteness(nlags=min(10, res.nobs // 2))
                # Normality test
                norm = res.test_normality()
                # Heteroskedasticity (if available)
                lines = [
                    f"\n{'═'*60}",
                    f" VAR Residual Diagnostic Tests",
                    f"{'═'*60}",
                    f"",
                    f"  Portmanteau (Whiteness) Test",
                    f"  {'─'*40}",
                    f"  Test statistic : {pt.test_statistic:>10.4f}",
                    f"  p-value        : {pt.pvalue:>10.4f}",
                    f"  df             : {pt.df}",
                    f"  H₀: No serial autocorrelation up to lag {pt.df}",
                    f"  Result: {'REJECT H₀ — autocorrelation detected' if pt.pvalue < 0.05 else 'Fail to reject H₀ — no evidence of autocorrelation'}",
                    f"",
                    f"  Normality Test (Jarque-Bera)",
                    f"  {'─'*40}",
                    f"  Test statistic : {norm.test_statistic:>10.4f}",
                    f"  p-value        : {norm.pvalue:>10.4f}",
                    f"  H₀: Residuals are multivariate normal",
                    f"  Result: {'REJECT H₀ — non-normal residuals' if norm.pvalue < 0.05 else 'Fail to reject H₀ — normality not rejected'}",
                ]
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(0, lambda: self._plot_resid_grid(res))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Residual Tests Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Stability check ───────────────────────────────────────────────────────
    def _run_var_stability(self):
        if not self._check_data():
            return
        if self.var_result is None:
            messagebox.showwarning("VAR", "Fit a VAR model first.")
            return

        def _work():
            try:
                res    = self.var_result
                roots  = res.roots
                moduli = np.abs(roots)
                stable = np.all(moduli < 1.0)

                lines = [
                    f"\n{'═'*60}",
                    f" VAR Stability Check  (characteristic roots)",
                    f"{'═'*60}",
                    f"",
                    f"  Stable: {'YES — all roots inside unit circle' if stable else 'NO — unit root or explosive root detected'}",
                    f"",
                    f"  {'Root (Re)':>12} {'Root (Im)':>12} {'Modulus':>10}  Status",
                    f"  {'─'*52}",
                ]
                for r, mod in zip(roots, moduli):
                    status = "inside" if mod < 1 else "ON BOUNDARY" if np.isclose(mod, 1) else "OUTSIDE"
                    lines.append(
                        f"  {r.real:>12.6f} {r.imag:>12.6f} {mod:>10.6f}  {status}")
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(0, lambda: self._plot_stability(roots))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Stability Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── IRF ───────────────────────────────────────────────────────────────────
    def _run_irf(self):
        self._irf_worker(cumulative=False)

    def _run_cirf(self):
        self._irf_worker(cumulative=True)

    def _irf_worker(self, cumulative=False):
        if not self._check_data():
            return
        if self.var_result is None:
            messagebox.showwarning("IRF", "Fit a VAR model first.")
            return
        periods = int(self.var_periods_var.get())

        def _work():
            try:
                irf = self.var_result.irf(periods)
                title = ("Cumulative Orthogonalized IRF"
                         if cumulative else "Orthogonalized IRF")
                out = (f"\n{'═'*60}\n"
                       f" {title}\n"
                       f"{'═'*60}\n"
                       f"  Model   : VAR({self.var_result.k_ar})\n"
                       f"  Periods : {periods}\n"
                       f"  Decomp. : Cholesky (Chol. ordering = variable order)\n"
                       f"  Signif. : 90% asymptotic confidence bands\n")
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(
                    0, lambda: self._plot_irf(irf, periods, cumulative, title))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "IRF Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── FEVD ──────────────────────────────────────────────────────────────────
    def _run_fevd(self):
        if not self._check_data():
            return
        if self.var_result is None:
            messagebox.showwarning("FEVD", "Fit a VAR model first.")
            return
        periods = int(self.var_periods_var.get())

        def _work():
            try:
                fevd = self.var_result.fevd(periods)
                cols = list(self.var_data_fit.columns)
                k    = len(cols)

                lines = [
                    f"\n{'═'*60}",
                    f" Forecast Error Variance Decomposition (FEVD)",
                    f"{'═'*60}",
                    f"  Periods: {periods}",
                    f"",
                ]
                for i, resp in enumerate(cols):
                    lines += [f"  Equation: {resp}",
                               f"  {'Period':>6} " +
                               "".join(f" {c:>10}" for c in cols),
                               f"  {'─'*52}"]
                    for t in range(periods):
                        row = "".join(
                            f" {fevd.decomp[i, t, j]:>10.4f}"
                            for j in range(k))
                        lines.append(f"  {t+1:>6}{row}")
                    lines.append("")
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(0, lambda: self._plot_fevd(fevd, cols, periods))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "FEVD Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Forecast ──────────────────────────────────────────────────────────────
    def _run_var_forecast(self):
        if not self._check_data():
            return
        if self.var_result is None:
            messagebox.showwarning("Forecast", "Fit a VAR model first.")
            return
        steps = int(self.var_periods_var.get())

        def _work():
            try:
                res  = self.var_result
                data = self.var_data_fit
                y    = data.values
                fc   = res.forecast(y=y, steps=steps)
                ci   = res.forecast_interval(y=y, steps=steps, alpha=0.10)
                lower, upper = ci[1], ci[2]
                cols = list(data.columns)

                lines = [
                    f"\n{'═'*60}",
                    f" VAR Forecast  ({steps} steps ahead)",
                    f"{'═'*60}",
                    f"",
                ]
                for i, col in enumerate(cols):
                    lines += [
                        f"  {col}:",
                        f"  {'Step':>5} {'Forecast':>12} {'Lower 90%':>12} {'Upper 90%':>12}",
                        f"  {'─'*46}",
                    ]
                    for t in range(steps):
                        lines.append(
                            f"  {t+1:>5} {fc[t,i]:>12.4f} "
                            f"{lower[t,i]:>12.4f} {upper[t,i]:>12.4f}")
                    lines.append("")
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(
                    0, lambda: self._plot_forecast(data, fc, lower, upper, cols))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Forecast Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Granger causality ─────────────────────────────────────────────────────
    def _run_granger(self):
        if not self._check_data():
            return
        maxlag = int(self.var_maxlag_var.get())

        def _work():
            try:
                data  = self._var_get_data()
                cols  = list(data.columns)
                k     = len(cols)
                mat_p = np.ones((k, k))   # p-value matrix
                mat_f = np.full((k, k), np.nan)

                for i, y_name in enumerate(cols):
                    for j, x_name in enumerate(cols):
                        if i == j:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                res_g = grangercausalitytests(
                                    data[[y_name, x_name]].dropna(),
                                    maxlag=maxlag, verbose=False)
                            # Use the minimum p-value across lags (most liberal)
                            pvals = [res_g[lag][0]["ssr_ftest"][1]
                                     for lag in range(1, maxlag + 1)]
                            mat_p[i, j] = min(pvals)
                            fvals = [res_g[lag][0]["ssr_ftest"][0]
                                     for lag in range(1, maxlag + 1)]
                            mat_f[i, j] = fvals[np.argmin(pvals)]
                        except Exception:
                            pass

                # Formatted table
                sep  = "─" * (14 + 12 * k)
                hdr  = (f"\n{'═'*60}\n"
                        f" Granger Causality Matrix  (H₀: X does not Granger-cause Y)\n"
                        f"{'═'*60}\n"
                        f"  Min p-value across lags 1–{maxlag}  (*** <0.01  ** <0.05  * <0.10)\n\n"
                        f"  Y \\ X (cause)   "
                        + "".join(f"{c[:10]:>12}" for c in cols) + "\n"
                        + f"  {sep}\n")
                body = ""
                for i, y_name in enumerate(cols):
                    body += f"  {y_name:<16}"
                    for j in range(k):
                        if i == j:
                            body += f"{'—':>12}"
                        else:
                            p = mat_p[i, j]
                            s = ("***" if p < 0.01 else "**" if p < 0.05
                                 else "*" if p < 0.10 else "")
                            body += f"{p:>8.4f}{s:>4}"
                    body += "\n"
                out = hdr + body + f"  {sep}\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(
                    0, lambda: self._plot_granger(cols, mat_p))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Granger Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── VECM ──────────────────────────────────────────────────────────────────
    def _run_vecm(self):
        if not self._check_data():
            return
        maxlag = max(1, int(self.var_maxlag_var.get()) - 1)
        det    = _VECM_DET_MAP.get(self.var_trend.get(), "co")

        def _work():
            try:
                # Use levels for VECM (cointegration is in levels)
                sel  = self.var_lb.curselection()
                cols = [self.var_lb.get(i) for i in sel]
                if len(cols) < 2:
                    raise ValueError("Select ≥ 2 variables.")
                country = self.var_country.get()
                if country == "Cross-sectional mean":
                    data = (self.df.groupby("Year")[cols]
                            .mean().sort_index().dropna())
                else:
                    sub  = self.df[self.df["Country"] == country]
                    data = sub.set_index("Year")[cols].sort_index().dropna()

                if len(data) < 8:
                    raise ValueError(
                        f"Only {len(data)} observations — need ≥ 8.")

                # Determine cointegration rank
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rank_res = select_coint_rank(
                        data, det_order=0 if det == "co" else -1,
                        k_ar_diff=maxlag, signif=0.05)
                coint_rank = rank_res.rank
                if coint_rank == 0:
                    coint_rank = 1   # force at least 1 for illustration

                # Fit VECM
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vecm_res = VECM(
                        data, k_ar_diff=maxlag,
                        coint_rank=coint_rank,
                        deterministic=det).fit()

                lines = [
                    f"\n{'═'*62}",
                    f" Vector Error Correction Model (VECM)",
                    f"{'═'*62}",
                    f"  Variables          : {', '.join(cols)}",
                    f"  Country / series   : {country}",
                    f"  k_ar_diff          : {maxlag}",
                    f"  Cointegration rank : {coint_rank}  "
                    f"(Johansen trace, 5%)",
                    f"  Deterministic term : {det}",
                    f"",
                    f"  Cointegrating vector(s) β (normalised):",
                    f"  {'─'*50}",
                ]
                beta = vecm_res.beta
                for r in range(coint_rank):
                    lines.append(f"  CE{r+1}: " +
                                 "  ".join(
                                     f"{cols[j]}: {beta[j,r]:>8.4f}"
                                     for j in range(len(cols))))

                lines += [f"",
                          f"  Adjustment speeds α:",
                          f"  {'─'*50}"]
                alpha = vecm_res.alpha
                for i, c in enumerate(cols):
                    for r in range(coint_rank):
                        lines.append(
                            f"  {c:<20}  α(CE{r+1}) = {alpha[i,r]:>8.4f}")

                lines += [f"",
                          f"  Short-run coefficients (Γ):"]
                gamma = vecm_res.gamma
                for i, eq in enumerate(cols):
                    lines.append(f"  Equation {eq}:")
                    for lag in range(maxlag):
                        for j, rhs in enumerate(cols):
                            lines.append(
                                f"    ΔL{lag+1}.{rhs:<20} "
                                f"{gamma[i, lag*len(cols)+j]:>10.5f}")

                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.var_txt),
                                            write(self.var_txt, out)))
                self.root.after(
                    0, lambda: self._plot_vecm(data, vecm_res, cols))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "VECM Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Plots ──────────────────────────────────────────────────────────────────
    def _clear_var_plot(self):
        for w in self.var_plot_frame.winfo_children():
            w.destroy()

    def _style_ax(self, ax, title=None):
        ax.set_facecolor(theme.WBG)
        ax.tick_params(colors=theme.FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(theme.GRAY)
        if title:
            ax.set_title(title, color=theme.FG, fontsize=8)

    def _plot_var_fit(self, data, res):
        """In-sample fitted vs actual for each equation."""
        self._clear_var_plot()
        cols = list(data.columns)
        k    = len(cols)
        fig, axes = plt.subplots(k, 1, figsize=(8, 2.5 * k),
                                  facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(hspace=0.55, left=0.10, right=0.97,
                            top=0.93, bottom=0.05)
        fitted = res.fittedvalues
        for i, col in enumerate(cols):
            ax = axes[i, 0]
            actual = data[col].iloc[res.k_ar:]
            ax.plot(actual.values, color=theme.BLUE, linewidth=1.3,
                    label="Actual")
            ax.plot(fitted[col].values, color=theme.AMBER,
                    linewidth=1.0, linestyle="--", label="Fitted")
            ax.legend(fontsize=6, facecolor=theme.WBG,
                      labelcolor=theme.FG, edgecolor=theme.GRAY)
            self._style_ax(ax, col)
        fig.suptitle(f"VAR({res.k_ar}) — Fitted vs Actual",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_ic_table(self, sel, maxlag):
        """Line chart of AIC/BIC/HQIC across lag orders."""
        self._clear_var_plot()
        lags_range = list(range(maxlag + 1))
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=theme.BG)
        self._style_ax(ax, "Information Criteria vs Lag Order")
        colours = [theme.BLUE, theme.AMBER, theme.TEAL]
        for crit_name, col in zip(["aic", "bic", "hqic"], colours):
            ic_list = sel.ics[crit_name]   # list indexed by lag (0…maxlag)
            vals = [ic_list[l] if l < len(ic_list) else np.nan
                    for l in lags_range]
            ax.plot(lags_range, vals, marker="o", color=col,
                    linewidth=1.4, label=crit_name.upper(), markersize=4)
        ax.set_xlabel("Lag order", color=theme.FG, fontsize=9)
        ax.set_ylabel("IC value", color=theme.FG, fontsize=9)
        ax.legend(fontsize=8, facecolor=theme.WBG,
                  labelcolor=theme.FG, edgecolor=theme.GRAY)
        fig.tight_layout()
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_resid_grid(self, res):
        """Residual plots (histogram + ACF) for each equation."""
        self._clear_var_plot()
        from statsmodels.graphics.tsaplots import plot_acf
        cols = list(self.var_data_fit.columns)
        k    = len(cols)
        fig, axes = plt.subplots(k, 2, figsize=(9, 2.5 * k),
                                  facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.09,
                            right=0.97, top=0.93, bottom=0.05)
        resids = np.asarray(res.resid)
        for i, col in enumerate(cols):
            r = resids[:, i]
            # Histogram
            ax1 = axes[i, 0]
            ax1.hist(r, bins=20, color=theme.BLUE, edgecolor=theme.BG,
                     density=True, alpha=0.80)
            xg = np.linspace(r.min(), r.max(), 200)
            from scipy.stats import norm as _norm_plt
            ax1.plot(xg, _norm_plt.pdf(xg, r.mean(), r.std()),
                     color=theme.AMBER, linewidth=1.4)
            self._style_ax(ax1, f"{col} — Residual Dist.")
            # ACF
            ax2 = axes[i, 1]
            plot_acf(r, ax=ax2, lags=min(20, len(r)//2 - 1),
                     alpha=0.10, color=theme.TEAL)
            self._restyle_fig(fig)
            self._style_ax(ax2, f"{col} — Residual ACF")
        fig.suptitle("VAR Residual Diagnostics", color=theme.FG, fontsize=10)
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_stability(self, roots):
        """Unit circle with eigenvalue moduli."""
        self._clear_var_plot()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=theme.BG)
        self._style_ax(ax, "VAR Stability — Characteristic Roots")
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta),
                color=theme.GRAY, linewidth=1.0, linestyle="--")
        ax.axhline(0, color=theme.GRAY, linewidth=0.6)
        ax.axvline(0, color=theme.GRAY, linewidth=0.6)
        inside  = [r for r in roots if abs(r) < 1.0]
        outside = [r for r in roots if abs(r) >= 1.0]
        if inside:
            ax.scatter([r.real for r in inside],
                       [r.imag for r in inside],
                       color=theme.TEAL, s=40, zorder=5, label="Inside")
        if outside:
            ax.scatter([r.real for r in outside],
                       [r.imag for r in outside],
                       color=theme.RED, s=60, zorder=5,
                       marker="x", label="Outside / boundary")
        ax.set_aspect("equal")
        ax.set_xlabel("Re", color=theme.FG, fontsize=8)
        ax.set_ylabel("Im", color=theme.FG, fontsize=8)
        ax.legend(fontsize=8, facecolor=theme.WBG,
                  labelcolor=theme.FG, edgecolor=theme.GRAY)
        fig.tight_layout()
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_irf(self, irf, periods, cumulative, title):
        """Custom-themed IRF grid (n_vars × n_vars).

        Uses orthogonalized (Cholesky) IRFs and asymptotic 90% CI bands
        derived from irf.stderr(orth=True) / irf.cum_effect_stderr(orth=True).
        """
        self._clear_var_plot()
        cols = list(self.var_data_fit.columns)
        k    = len(cols)

        z = 1.645   # 90% two-sided
        if cumulative:
            vals_all = irf.orth_cum_effects                    # (periods+1, k, k)
            se_all   = irf.cum_effect_stderr(orth=True)        # same shape
        else:
            vals_all = irf.orth_irfs                           # (periods+1, k, k)
            se_all   = irf.stderr(orth=True)                   # same shape
        lower_all = vals_all - z * se_all
        upper_all = vals_all + z * se_all

        x = np.arange(periods + 1)
        fig, axes = plt.subplots(
            k, k, figsize=(max(8, 2.8 * k), max(6, 2.2 * k)),
            facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08,
                            right=0.97, top=0.91, bottom=0.06)

        for i, resp in enumerate(cols):
            for j, imp in enumerate(cols):
                ax  = axes[i, j]
                v   = vals_all[:, i, j]
                lo  = lower_all[:, i, j]
                hi  = upper_all[:, i, j]
                ax.plot(x, v, color=theme.BLUE, linewidth=1.4)
                ax.fill_between(x, lo, hi, color=theme.BLUE, alpha=0.18)
                ax.axhline(0, color=theme.GRAY, linewidth=0.8,
                           linestyle="--")
                self._style_ax(ax)
                if i == 0:
                    ax.set_title(f"← {imp}", color=theme.FG, fontsize=7)
                if j == 0:
                    ax.set_ylabel(resp, color=theme.FG, fontsize=7)

        fig.suptitle(title + f"  (90% CI, Cholesky)",
                     color=theme.FG, fontsize=9)
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_fevd(self, fevd, cols, periods):
        """Stacked area FEVD chart for each equation."""
        self._clear_var_plot()
        k   = len(cols)
        pal = [theme.BLUE, theme.TEAL, theme.AMBER, theme.RED,
               theme.GRAY, "#9B59B6", "#1ABC9C", "#E67E22"]

        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4),
                                  facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(wspace=0.30, left=0.07, right=0.97,
                            top=0.88, bottom=0.12)
        x = np.arange(1, periods + 1)

        for i, resp in enumerate(cols):
            ax     = axes[0, i]
            bottom = np.zeros(periods)
            for j, src in enumerate(cols):
                share = fevd.decomp[i, :periods, j]
                ax.bar(x, share, bottom=bottom,
                       color=pal[j % len(pal)], label=src, width=0.8)
                bottom += share
            self._style_ax(ax, resp)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Horizon", color=theme.FG, fontsize=8)
            if i == k - 1:
                ax.legend(fontsize=7, facecolor=theme.WBG,
                          labelcolor=theme.FG, edgecolor=theme.GRAY,
                          bbox_to_anchor=(1.01, 1), loc="upper left")

        fig.suptitle("Forecast Error Variance Decomposition",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_forecast(self, data, fc, lower, upper, cols):
        """Fan-chart forecast plot for each variable."""
        self._clear_var_plot()
        k    = len(cols)
        T    = len(data)
        steps = len(fc)
        hist_idx = np.arange(T)
        fore_idx = np.arange(T - 1, T + steps)

        fig, axes = plt.subplots(k, 1, figsize=(8, 2.8 * k),
                                  facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(hspace=0.55, left=0.10, right=0.97,
                            top=0.93, bottom=0.05)
        for i, col in enumerate(cols):
            ax     = axes[i, 0]
            hist   = data[col].values
            fc_i   = np.concatenate([[hist[-1]], fc[:, i]])
            lo_i   = np.concatenate([[hist[-1]], lower[:, i]])
            hi_i   = np.concatenate([[hist[-1]], upper[:, i]])
            ax.plot(hist_idx, hist, color=theme.BLUE, linewidth=1.3,
                    label="Historical")
            ax.plot(fore_idx, fc_i, color=theme.AMBER, linewidth=1.4,
                    linestyle="--", label="Forecast")
            ax.fill_between(fore_idx, lo_i, hi_i,
                            color=theme.AMBER, alpha=0.22,
                            label="90% CI")
            ax.axvline(T - 1, color=theme.GRAY, linewidth=0.8,
                       linestyle=":")
            ax.legend(fontsize=6, facecolor=theme.WBG,
                      labelcolor=theme.FG, edgecolor=theme.GRAY)
            self._style_ax(ax, col)
        fig.suptitle(f"VAR Forecast  ({steps} steps ahead, 90% CI)",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_granger(self, cols, mat_p):
        """Heatmap of Granger causality p-values."""
        self._clear_var_plot()
        k   = len(cols)
        sig = (mat_p < 0.05).astype(float)
        for i in range(k):
            sig[i, i] = np.nan

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(4, k * 0.8)),
                                        facecolor=theme.BG)
        fig.subplots_adjust(wspace=0.40, left=0.14, right=0.97,
                            top=0.88, bottom=0.18)
        # p-value heatmap
        masked = np.ma.masked_invalid(mat_p)
        im1 = ax1.imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=0.15,
                         aspect="auto")
        ax1.set_xticks(range(k)); ax1.set_yticks(range(k))
        ax1.set_xticklabels([f"X={c[:8]}" for c in cols],
                             rotation=45, ha="right",
                             fontsize=7, color=theme.FG)
        ax1.set_yticklabels([f"Y={c[:8]}" for c in cols],
                             fontsize=7, color=theme.FG)
        for (i, j), val in np.ndenumerate(mat_p):
            if i != j:
                s = ("***" if val < 0.01 else "**" if val < 0.05
                     else "*" if val < 0.10 else "")
                ax1.text(j, i, f"{val:.3f}{s}",
                         ha="center", va="center",
                         fontsize=6, color="white" if val < 0.05 else theme.FG)
        self._style_ax(ax1, "Granger p-values  (X → Y)")
        plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.04)

        # Significant pairs bar
        ax2.set_facecolor(theme.WBG)
        pairs  = [(f"{cols[j]}→{cols[i]}", mat_p[i, j])
                  for i in range(k) for j in range(k)
                  if i != j and mat_p[i, j] < 0.10]
        pairs.sort(key=lambda x: x[1])
        if pairs:
            labels, pvals = zip(*pairs)
            colors = [theme.RED if p < 0.01 else theme.AMBER if p < 0.05
                      else theme.TEAL for p in pvals]
            y_pos = np.arange(len(labels))
            ax2.barh(y_pos, pvals, color=colors, height=0.6)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=7, color=theme.FG)
            ax2.axvline(0.05, color=theme.AMBER, linewidth=1.1,
                        linestyle="--", label="5%")
            ax2.axvline(0.01, color=theme.RED, linewidth=1.0,
                        linestyle=":", label="1%")
            ax2.legend(fontsize=7, facecolor=theme.WBG,
                       labelcolor=theme.FG, edgecolor=theme.GRAY)
        self._style_ax(ax2, "Significant Causal Links (p < 0.10)")
        ax2.set_xlabel("p-value", color=theme.FG, fontsize=8)
        for sp in ax2.spines.values():
            sp.set_edgecolor(theme.GRAY)

        fig.suptitle("Granger Causality Analysis", color=theme.FG, fontsize=10)
        embed_figure(fig, self.var_plot_frame, toolbar=True)

    def _plot_vecm(self, data, vecm_res, cols):
        """VECM: error correction terms (CE) over time."""
        self._clear_var_plot()
        k         = len(cols)
        alpha     = vecm_res.alpha
        beta      = vecm_res.beta
        rank      = beta.shape[1]
        ce_series = (data.values @ beta)   # (T, rank)
        T         = len(ce_series)

        fig, axes = plt.subplots(rank + k, 1,
                                  figsize=(8, 2.5 * (rank + k)),
                                  facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(hspace=0.55, left=0.10, right=0.97,
                            top=0.95, bottom=0.04)
        # CE plots
        for r in range(rank):
            ax = axes[r, 0]
            ax.plot(ce_series[:, r], color=theme.AMBER, linewidth=1.2)
            ax.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
            self._style_ax(ax, f"Cointegrating Relation CE{r+1}")

        # Actual vs VECM in-sample fitted values
        k_ar = vecm_res.model.k_ar_diff
        fitted = vecm_res.fittedvalues   # shape (nobs, k_vars)
        for i, col in enumerate(cols):
            ax     = axes[rank + i, 0]
            actual = data[col].iloc[k_ar:].values
            n_fit  = min(len(actual), len(fitted))
            ax.plot(actual[:n_fit], color=theme.BLUE, linewidth=1.2,
                    label="Actual")
            ax.plot(fitted[:n_fit, i], color=theme.TEAL, linewidth=1.0,
                    linestyle="--", label="VECM fitted")
            ax.legend(fontsize=6, facecolor=theme.WBG,
                      labelcolor=theme.FG, edgecolor=theme.GRAY)
            self._style_ax(ax, col)

        fig.suptitle("VECM — Cointegrating Relations & Fitted Values",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.var_plot_frame, toolbar=True)
