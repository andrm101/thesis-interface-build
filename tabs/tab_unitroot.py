"""
tabs/tab_unitroot.py  —  Unit Root & Cointegration tab.

Tests provided
──────────────
  · ADF  (Augmented Dickey-Fuller)              statsmodels
  · KPSS (Kwiatkowski-Phillips-Schmidt-Shin)    statsmodels
  · PP   (Phillips-Perron)                      arch  (optional)
  · IPS  (Im-Pesaran-Shin panel unit root)      manual implementation
  · Engle-Granger cointegration                 statsmodels
  · Johansen cointegration (trace + max-λ)      statsmodels
"""

import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, messagebox

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import norm as _norm

try:
    from arch.unitroot import PhillipsPerron as _PP
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False

import panel_tests
import theme
from helpers import make_text, write, clear_txt, embed_figure


class UnitRootTabMixin:

    # ── Tab builder ───────────────────────────────────────────────────────────
    def _tab_unitroot(self, parent):
        # ── Row 1: variable / trend / lags ───────────────────────────────────
        r1 = ttk.Frame(parent)
        r1.pack(fill=tk.X, padx=12, pady=(8, 2))

        ttk.Label(r1, text="Variable:").pack(side=tk.LEFT)
        self.ur_var = ttk.Combobox(r1, width=22, state="readonly")
        self.ur_var.pack(side=tk.LEFT, padx=(4, 14))

        ttk.Label(r1, text="Trend:").pack(side=tk.LEFT)
        self.ur_trend = ttk.Combobox(
            r1, width=20, state="readonly",
            values=["Constant", "Constant + Trend", "No constant"])
        self.ur_trend.current(0)
        self.ur_trend.pack(side=tk.LEFT, padx=(4, 14))

        ttk.Label(r1, text="Max lags:").pack(side=tk.LEFT)
        self.ur_maxlag_var = tk.StringVar(value="4")
        sp_lag = tk.Spinbox(r1, from_=0, to=12, width=4,
                            textvariable=self.ur_maxlag_var,
                            bg=theme.WBG, fg=theme.FG,
                            insertbackground=theme.FG,
                            buttonbackground=theme.WBG)
        sp_lag.pack(side=tk.LEFT, padx=(4, 14))
        self._tk_spinboxes.append(sp_lag)

        ttk.Label(r1, text="Lag selection:").pack(side=tk.LEFT)
        self.ur_lagsel = ttk.Combobox(
            r1, width=8, state="readonly",
            values=["AIC", "BIC", "t-stat", "Fixed"])
        self.ur_lagsel.current(0)
        self.ur_lagsel.pack(side=tk.LEFT, padx=(4, 0))

        # ── Row 2: unit root test buttons ─────────────────────────────────────
        r2 = ttk.Frame(parent)
        r2.pack(fill=tk.X, padx=12, pady=(4, 2))

        for label, cmd in [
            ("ADF Test",            self._run_adf),
            ("KPSS Test",           self._run_kpss),
            ("PP Test",             self._run_pp),
            ("ADF — All Variables", self._run_adf_all),
            ("IPS Panel Test",      self._run_ips),
            ("CIPS (Pesaran)",      self._run_cips),
            ("CIPS — All",          self._run_cips_all),
        ]:
            ttk.Button(r2, text=label, command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Row 3: cointegration controls ────────────────────────────────────
        r3 = ttk.Frame(parent)
        r3.pack(fill=tk.X, padx=12, pady=(10, 2))

        coint_hdr = ttk.Label(r3, text="── Cointegration ──",
                              foreground=theme.TEAL)
        coint_hdr.pack(side=tk.LEFT, padx=(0, 10))
        self._special_labels.append((coint_hdr, "TEAL"))

        ttk.Label(r3, text="Var 1:").pack(side=tk.LEFT)
        self.ur_coint1 = ttk.Combobox(r3, width=20, state="readonly")
        self.ur_coint1.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(r3, text="Var 2:").pack(side=tk.LEFT)
        self.ur_coint2 = ttk.Combobox(r3, width=20, state="readonly")
        self.ur_coint2.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(r3, text="Det. order:").pack(side=tk.LEFT)
        self.ur_det = ttk.Combobox(
            r3, width=20, state="readonly",
            values=["No constant (-1)", "Constant (0)", "Constant + trend (1)"])
        self.ur_det.current(1)
        self.ur_det.pack(side=tk.LEFT, padx=(4, 12))

        for label, cmd in [
            ("Engle-Granger",       self._run_eg),
            ("Johansen",            self._run_johansen),
            ("Johansen — All Pairs",self._run_johansen_all),
        ]:
            ttk.Button(r3, text=label, command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Row 4: Johansen variable listbox ─────────────────────────────────
        r4 = ttk.Frame(parent)
        r4.pack(fill=tk.X, padx=12, pady=(4, 4))

        ttk.Label(r4, text="Johansen variables (select ≥ 2):").pack(
            side=tk.LEFT, padx=(0, 8))
        self.ur_joh_lb = tk.Listbox(
            r4, selectmode=tk.MULTIPLE, height=3, width=42,
            bg=theme.WBG, fg=theme.FG,
            selectbackground=theme.BLUE,
            selectforeground=theme.WFG,
            exportselection=False)
        self.ur_joh_lb.pack(side=tk.LEFT)
        sb_joh = ttk.Scrollbar(r4, orient=tk.VERTICAL,
                               command=self.ur_joh_lb.yview)
        sb_joh.pack(side=tk.LEFT, fill=tk.Y)
        self.ur_joh_lb.configure(yscrollcommand=sb_joh.set)

        # ── Paned results area ────────────────────────────────────────────────
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.ur_txt        = make_text(left, height=30, copy_root=self.root)
        self.ur_plot_frame = right

        self._refresh_ur_vars()

    # ── Variable refresh ──────────────────────────────────────────────────────
    def _refresh_ur_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        for cb in (self.ur_var, self.ur_coint1, self.ur_coint2):
            cb["values"] = num_cols
            if num_cols:
                cb.current(0)
        # Default the cointegration pair to two *different* series.
        if len(num_cols) > 1:
            self.ur_coint2.current(1)
        self.ur_joh_lb.delete(0, tk.END)
        for c in num_cols:
            self.ur_joh_lb.insert(tk.END, c)

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _trend_code(label):
        return {"Constant": "c",
                "Constant + Trend": "ct",
                "No constant": "n"}.get(label, "c")

    @staticmethod
    def _det_order(label):
        return {"No constant (-1)": -1,
                "Constant (0)": 0,
                "Constant + trend (1)": 1}.get(label, 0)

    def _ur_series(self, col):
        """Pooled series for a single column, NaNs dropped."""
        return self.df[col].dropna()

    def _ur_panel_series(self, col):
        """Return {entity: series} for panel tests (≥ 5 obs per entity)."""
        out = {}
        for country, grp in self.df.groupby("Country"):
            s = grp.set_index("Year")[col].dropna()
            if len(s) >= 5:
                out[country] = s
        return out

    def _stars(self, p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    def _adf_fmt(self, col, res, trend_label):
        stat, pval, lags_used, nobs, crit, ic = res
        return (
            f"\n{'─'*54}\n"
            f" ADF Test  —  {col}\n"
            f"{'─'*54}\n"
            f"  Trend specification : {trend_label}\n"
            f"  Lags used           : {lags_used}  (IC best: {ic:.4f})\n"
            f"  Observations        : {nobs}\n\n"
            f"  ADF statistic       : {stat:>10.4f}\n"
            f"  p-value             : {pval:>10.4f}  {self._stars(pval)}\n\n"
            f"  Critical values:\n"
            f"    1%  : {crit['1%']:>8.4f}\n"
            f"    5%  : {crit['5%']:>8.4f}\n"
            f"   10%  : {crit['10%']:>8.4f}\n\n"
            f"  Conclusion: "
            f"{'Reject H₀ — series likely stationary I(0)' if pval < 0.05 else 'Fail to reject H₀ — series likely non-stationary I(1)'}\n"
        )

    # ── Plot utility ──────────────────────────────────────────────────────────
    def _clear_ur_plot(self):
        for w in self.ur_plot_frame.winfo_children():
            w.destroy()

    def _style_ax(self, ax, title=None):
        ax.set_facecolor(theme.WBG)
        ax.tick_params(colors=theme.FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(theme.GRAY)
        if title:
            ax.set_title(title, color=theme.FG, fontsize=9)

    # ── ADF (single variable) ─────────────────────────────────────────────────
    def _run_adf(self):
        if not self._check_data():
            return
        col        = self.ur_var.get()
        trend      = self._trend_code(self.ur_trend.get())
        trend_lbl  = self.ur_trend.get()
        maxlag     = int(self.ur_maxlag_var.get())
        lag_method = self.ur_lagsel.get()
        autolag    = None if lag_method == "Fixed" else lag_method

        def _work():
            try:
                s   = self._ur_series(col)
                res = adfuller(s, maxlag=maxlag, regression=trend, autolag=autolag)
                out = self._adf_fmt(col, res, trend_lbl)
                self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                            write(self.ur_txt, out)))
                self.root.after(0, lambda: self._plot_level_diff(col, s))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("ADF Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── KPSS ─────────────────────────────────────────────────────────────────
    def _run_kpss(self):
        if not self._check_data():
            return
        col       = self.ur_var.get()
        trend_lbl = self.ur_trend.get()
        reg       = "ct" if "Trend" in trend_lbl else "c"

        def _work():
            try:
                s = self._ur_series(col)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, pval, lags, crit = kpss(s, regression=reg, nlags="auto")
                out = (
                    f"\n{'─'*54}\n"
                    f" KPSS Test  —  {col}\n"
                    f"{'─'*54}\n"
                    f"  Trend specification : {trend_lbl}\n"
                    f"  Lags (Newey-West)   : {lags}\n\n"
                    f"  KPSS statistic      : {stat:>10.4f}\n"
                    f"  p-value (interp.)   : {pval:>10.4f}  {self._stars(pval)}\n\n"
                    f"  Critical values:\n"
                    f"   10%  : {crit['10%']:>8.4f}\n"
                    f"    5%  : {crit['5%']:>8.4f}\n"
                    f"  2.5%  : {crit['2.5%']:>8.4f}\n"
                    f"    1%  : {crit['1%']:>8.4f}\n\n"
                    f"  H₀: Series is stationary\n"
                    f"  Conclusion: "
                    f"{'Reject H₀ — series likely non-stationary I(1)' if pval < 0.05 else 'Fail to reject H₀ — series likely stationary I(0)'}\n"
                )
                self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                            write(self.ur_txt, out)))
                self.root.after(0, lambda: self._plot_level_diff(col, s))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("KPSS Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Phillips-Perron ───────────────────────────────────────────────────────
    def _run_pp(self):
        if not self._check_data():
            return
        col       = self.ur_var.get()
        trend_lbl = self.ur_trend.get()

        def _work():
            try:
                s = self._ur_series(col).values
                if not _HAS_ARCH:
                    out = (
                        "\n  Phillips-Perron Test requires the 'arch' package.\n"
                        "  Install with:  pip install arch\n"
                    )
                    self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                                write(self.ur_txt, out)))
                    return

                trend_map = {"Constant": "c",
                             "Constant + Trend": "ct",
                             "No constant": "n"}
                pp   = _PP(s, trend=trend_map.get(trend_lbl, "c"))
                stat = pp.stat
                pval = pp.pvalue
                lags = pp.lags
                crit = pp.critical_values
                out  = (
                    f"\n{'─'*54}\n"
                    f" Phillips-Perron Test  —  {col}\n"
                    f"{'─'*54}\n"
                    f"  Trend specification : {trend_lbl}\n"
                    f"  Bandwidth (lags)    : {lags}\n\n"
                    f"  PP statistic        : {stat:>10.4f}\n"
                    f"  p-value             : {pval:>10.4f}  {self._stars(pval)}\n\n"
                    f"  Critical values:\n"
                    f"    1%  : {crit['1%']:>8.4f}\n"
                    f"    5%  : {crit['5%']:>8.4f}\n"
                    f"   10%  : {crit['10%']:>8.4f}\n\n"
                    f"  Conclusion: "
                    f"{'Reject H₀ — likely I(0)' if pval < 0.05 else 'Fail to reject H₀ — likely I(1)'}\n"
                )
                self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                            write(self.ur_txt, out)))
                self.root.after(0, lambda: self._plot_level_diff(
                    col, self._ur_series(col)))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("PP Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── ADF — all variables ───────────────────────────────────────────────────
    def _run_adf_all(self):
        if not self._check_data():
            return
        trend     = self._trend_code(self.ur_trend.get())
        trend_lbl = self.ur_trend.get()
        maxlag    = int(self.ur_maxlag_var.get())
        lag_meth  = self.ur_lagsel.get()
        autolag   = None if lag_meth == "Fixed" else lag_meth
        num_cols  = [c for c in self.df.select_dtypes("number").columns
                     if c != "Year"]

        def _work():
            rows         = []
            detail_parts = []
            for col in num_cols:
                try:
                    s   = self._ur_series(col)
                    res = adfuller(s, maxlag=maxlag, regression=trend,
                                  autolag=autolag)
                    stat, pval, lag, nobs, crit, _ = res
                    order = "I(0)" if pval < 0.05 else "I(1)"
                    rows.append((col, stat, pval, order, lag))
                    detail_parts.append(self._adf_fmt(col, res, trend_lbl))
                except Exception as e:
                    rows.append((col, float("nan"), float("nan"), "ERR", 0))

            # Summary table
            sep = "─" * 60
            hdr = (f"\n{'═'*60}\n"
                   f" ADF Summary — All Variables  ({trend_lbl})\n"
                   f"{'═'*60}\n"
                   f"  {'Variable':<26} {'ADF stat':>10} {'p-val':>8}  {'Sig':>4}  Order  Lags\n"
                   f"  {sep}\n")
            body = "".join(
                f"  {c:<26} {st:>10.4f} {pv:>8.4f}  {self._stars(pv):>4}  {ord_:>4}   {lg}\n"
                for c, st, pv, ord_, lg in rows)
            footer = f"  {sep}\n  *** p<0.01  ** p<0.05  * p<0.10\n"
            out = hdr + body + footer + "\n" + "".join(detail_parts)

            self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                        write(self.ur_txt, out)))
            self.root.after(0, lambda: self._plot_adf_summary(rows))

        threading.Thread(target=_work, daemon=True).start()

    # ── IPS panel unit root ───────────────────────────────────────────────────
    def _run_ips(self):
        """Im-Pesaran-Shin (2003) panel unit root test.

        Standardisation moments from IPS (2003) Table 3 (approximated
        response-surface midpoints for unbalanced panels).
        """
        if not self._check_data():
            return
        col       = self.ur_var.get()
        trend     = self._trend_code(self.ur_trend.get())
        trend_lbl = self.ur_trend.get()
        maxlag    = int(self.ur_maxlag_var.get())
        lag_meth  = self.ur_lagsel.get()
        autolag   = None if lag_meth == "Fixed" else lag_meth

        def _work():
            try:
                entities = self._ur_panel_series(col)
                N = len(entities)
                if N < 2:
                    raise ValueError("Need ≥ 2 entities with ≥ 5 obs each.")

                t_stats, T_is, detail = [], [], []
                for entity, s in entities.items():
                    res  = adfuller(s, maxlag=maxlag, regression=trend,
                                    autolag=autolag)
                    t_i  = res[0]
                    T_i  = res[3]
                    t_stats.append(t_i)
                    T_is.append(T_i)
                    detail.append(
                        f"  {entity:<28} {t_i:>9.4f}  {res[1]:>8.4f}  "
                        f"{self._stars(res[1]):>3}  T={T_i}")

                t_bar = np.mean(t_stats)
                T_bar = np.mean(T_is)

                # IPS (2003) Table 3 standardisation moments
                moments = {"c": (-1.530, 0.470),
                           "ct": (-2.230, 0.589),
                           "n":  (-1.000, 0.420)}
                E_t, Var_t = moments.get(trend, (-1.530, 0.470))

                W_bar = np.sqrt(N) * (t_bar - E_t) / np.sqrt(Var_t)
                pval  = _norm.cdf(W_bar)     # one-sided lower tail

                out = (
                    f"\n{'═'*60}\n"
                    f" Im-Pesaran-Shin Panel Unit Root Test  —  {col}\n"
                    f"{'═'*60}\n"
                    f"  Trend specification  : {trend_lbl}\n"
                    f"  Number of entities   : {N}\n"
                    f"  Avg. time obs. (T̄)   : {T_bar:.1f}\n\n"
                    f"  t-bar statistic      : {t_bar:>10.4f}\n"
                    f"  W-bar (standardised) : {W_bar:>10.4f}\n"
                    f"  p-value (one-sided)  : {pval:>10.4f}  {self._stars(pval)}\n\n"
                    f"  H₀: All panels contain a unit root\n"
                    f"  Conclusion: "
                    f"{'Reject H₀ — at least some panels stationary' if pval < 0.05 else 'Fail to reject H₀ — all panels non-stationary I(1)'}\n\n"
                    f"  Individual ADF results:\n"
                    f"  {'Entity':<28} {'t-stat':>9}  {'p-val':>8}  Sig  T\n"
                    f"  {'─'*56}\n"
                    + "\n".join(detail) + "\n"
                )
                self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                            write(self.ur_txt, out)))
                self.root.after(0, lambda: self._plot_ips(
                    col, entities, t_stats, t_bar, W_bar))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("IPS Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Engle-Granger ─────────────────────────────────────────────────────────
    def _run_eg(self):
        if not self._check_data():
            return
        col1      = self.ur_coint1.get()
        col2      = self.ur_coint2.get()
        trend_lbl = self.ur_trend.get()
        trend     = self._trend_code(trend_lbl)
        if not col1 or not col2 or col1 == col2:
            messagebox.showwarning(
                "Engle-Granger", "Pick two different variables to test.")
            return

        def _work():
            try:
                df_c = self.df[[col1, col2]].dropna()
                y, x = df_c[col1].values, df_c[col2].values
                tau, pval, crit = coint(y, x, trend=trend)
                out = (
                    f"\n{'═'*60}\n"
                    f" Engle-Granger Cointegration Test\n"
                    f"{'═'*60}\n"
                    f"  Variable 1   : {col1}\n"
                    f"  Variable 2   : {col2}\n"
                    f"  Trend        : {trend_lbl}\n"
                    f"  Observations : {len(y)}\n\n"
                    f"  τ-statistic  : {tau:>10.4f}\n"
                    f"  p-value      : {pval:>10.4f}  {self._stars(pval)}\n\n"
                    f"  Critical values:\n"
                    f"    1%  : {crit[0]:>8.4f}\n"
                    f"    5%  : {crit[1]:>8.4f}\n"
                    f"   10%  : {crit[2]:>8.4f}\n\n"
                    f"  H₀: No cointegration\n"
                    f"  Conclusion: "
                    f"{'Reject H₀ — series appear cointegrated' if pval < 0.05 else 'Fail to reject H₀ — no cointegration found'}\n"
                )
                self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                            write(self.ur_txt, out)))
                self.root.after(0, lambda: self._plot_eg(col1, col2, df_c))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("EG Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ── Johansen ──────────────────────────────────────────────────────────────
    def _run_johansen(self):
        if not self._check_data():
            return
        sel  = self.ur_joh_lb.curselection()
        cols = [self.ur_joh_lb.get(i) for i in sel]
        if len(cols) < 2:
            messagebox.showwarning(
                "Johansen", "Select ≥ 2 variables in the Johansen list.")
            return
        det    = self._det_order(self.ur_det.get())
        k_diff = max(1, int(self.ur_maxlag_var.get()) - 1)

        def _work():
            try:
                data = self.df[cols].dropna().values
                res  = coint_johansen(data, det, k_diff)
                out  = self._johansen_fmt(cols, res, det)
                self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                            write(self.ur_txt, out)))
                self.root.after(0, lambda: self._plot_johansen(cols, res))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Johansen Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _johansen_fmt(self, cols, res, det):
        n       = len(cols)
        ev      = res.eig
        tr_stat = res.lr1
        tr_cv   = res.cvt       # (n, 3)  at 90%, 95%, 99%
        mx_stat = res.lr2
        mx_cv   = res.cvm

        det_lbl = ["no constant", "constant", "constant + trend"][det + 1]
        lines   = [
            f"\n{'═'*66}",
            f" Johansen Cointegration Test",
            f"{'═'*66}",
            f"  Variables    : {', '.join(cols)}",
            f"  Det. order   : {det}  ({det_lbl})",
            f"",
            f"  Trace Test  (H₀: rank ≤ r)",
            f"  {'r':<4} {'Trace':>10} {'90%CV':>8} {'95%CV':>8} {'99%CV':>8}  Sig",
            f"  {'─'*54}",
        ]
        for r in range(n):
            sig = ("***" if tr_stat[r] > tr_cv[r, 2]
                   else "**" if tr_stat[r] > tr_cv[r, 1]
                   else "*"  if tr_stat[r] > tr_cv[r, 0]
                   else "")
            lines.append(f"  {r:<4} {tr_stat[r]:>10.4f} {tr_cv[r,0]:>8.4f} "
                         f"{tr_cv[r,1]:>8.4f} {tr_cv[r,2]:>8.4f}  {sig}")

        lines += [
            f"",
            f"  Max Eigenvalue Test  (H₀: rank = r  vs  H₁: rank = r+1)",
            f"  {'r':<4} {'Max-ev':>10} {'90%CV':>8} {'95%CV':>8} {'99%CV':>8}  Sig",
            f"  {'─'*54}",
        ]
        for r in range(n):
            sig = ("***" if mx_stat[r] > mx_cv[r, 2]
                   else "**" if mx_stat[r] > mx_cv[r, 1]
                   else "*"  if mx_stat[r] > mx_cv[r, 0]
                   else "")
            lines.append(f"  {r:<4} {mx_stat[r]:>10.4f} {mx_cv[r,0]:>8.4f} "
                         f"{mx_cv[r,1]:>8.4f} {mx_cv[r,2]:>8.4f}  {sig}")

        lines += [f"", f"  Eigenvalues:"]
        for i, v in enumerate(ev):
            lines.append(f"    λ{i+1} = {v:.6f}")
        lines.append("\n  *** 1%  ** 5%  * 10%  (rejection of H₀)")
        return "\n".join(lines) + "\n"

    # ── Johansen all pairs ────────────────────────────────────────────────────
    def _run_johansen_all(self):
        if not self._check_data():
            return
        det      = self._det_order(self.ur_det.get())
        k_diff   = max(1, int(self.ur_maxlag_var.get()) - 1)
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]

        def _work():
            rows = []
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i + 1:]:
                    try:
                        data = self.df[[c1, c2]].dropna().values
                        res  = coint_johansen(data, det, k_diff)
                        # 1 if trace rejects at 5%, else 0
                        r1   = int(res.lr1[0] > res.cvt[0, 1])
                        rows.append((c1, c2, res.lr1[0], r1))
                    except Exception:
                        rows.append((c1, c2, float("nan"), 0))

            sep = "─" * 60
            hdr = (f"\n{'═'*60}\n"
                   f" Johansen All-Pairs Summary  (trace stat, 5% CV)\n"
                   f"{'═'*60}\n"
                   f"  {'Var 1':<22} {'Var 2':<22} {'Trace':>8}  Coint?\n"
                   f"  {sep}\n")
            body = "".join(
                f"  {c1:<22} {c2:<22} {tr:>8.4f}  {'Yes ***' if r else 'No'}\n"
                for c1, c2, tr, r in rows)
            out = hdr + body + f"  {sep}\n"
            self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                        write(self.ur_txt, out)))
            self.root.after(0, lambda: self._plot_coint_heatmap(num_cols, rows))

        threading.Thread(target=_work, daemon=True).start()

    # ── Plots ──────────────────────────────────────────────────────────────────
    def _plot_level_diff(self, col, s):
        """Level and first-difference of a series."""
        self._clear_ur_plot()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                        facecolor=theme.BG)
        fig.subplots_adjust(hspace=0.48, left=0.10, right=0.97,
                            top=0.90, bottom=0.10)

        vals = s.values
        ax1.plot(vals, color=theme.BLUE, linewidth=1.3)
        self._style_ax(ax1, f"{col}  — Level")

        ds = np.diff(vals)
        ax2.plot(ds, color=theme.TEAL, linewidth=1.1)
        ax2.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        self._style_ax(ax2, f"Δ{col}  — First Difference")

        fig.suptitle("Level vs. First Difference", color=theme.FG, fontsize=10)
        embed_figure(fig, self.ur_plot_frame, toolbar=True)

    def _plot_adf_summary(self, rows):
        """Horizontal bar chart of ADF statistics for all variables."""
        self._clear_ur_plot()
        cols   = [r[0] for r in rows]
        stats  = [r[1] for r in rows]
        colors = [theme.TEAL if r[3] == "I(0)" else theme.RED for r in rows]

        fig, ax = plt.subplots(
            figsize=(7, max(3, len(cols) * 0.38 + 1.2)),
            facecolor=theme.BG)
        self._style_ax(ax, "ADF Test — All Variables")

        y_pos = np.arange(len(cols))
        ax.barh(y_pos, stats, color=colors, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cols, fontsize=8, color=theme.FG)
        ax.axvline(-1.95, color=theme.AMBER, linewidth=1.3,
                   linestyle="--", label="5% CV (≈ −1.95)")
        ax.axvline(-2.86, color=theme.RED, linewidth=1.0,
                   linestyle=":", label="1% CV (≈ −2.86)")
        ax.set_xlabel("ADF t-statistic", color=theme.FG, fontsize=9)
        ax.legend(fontsize=8, facecolor=theme.WBG,
                  labelcolor=theme.FG, edgecolor=theme.GRAY)
        # Legend patches for I(0)/I(1)
        from matplotlib.patches import Patch
        leg_extra = [Patch(color=theme.TEAL, label="I(0) — stationary"),
                     Patch(color=theme.RED,  label="I(1) — unit root")]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + leg_extra,
                  labels=ax.get_legend_handles_labels()[1]
                  + ["I(0) — stationary", "I(1) — unit root"],
                  fontsize=7, facecolor=theme.WBG,
                  labelcolor=theme.FG, edgecolor=theme.GRAY)
        fig.tight_layout()
        embed_figure(fig, self.ur_plot_frame, toolbar=True)

    def _plot_ips(self, col, entities, t_stats, t_bar, W_bar):
        """Individual ADF bars + W-bar on standard-normal density."""
        self._clear_ur_plot()
        names = list(entities.keys())
        n     = len(names)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, max(3.5, n * 0.30 + 1.5)),
            facecolor=theme.BG)
        fig.subplots_adjust(wspace=0.35, left=0.22, right=0.97,
                            top=0.88, bottom=0.10)

        # Left: individual t-stats
        y_pos  = np.arange(n)
        colors = [theme.TEAL if t < -1.95 else theme.RED for t in t_stats]
        ax1.barh(y_pos, t_stats, color=colors, height=0.6)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=7, color=theme.FG)
        ax1.axvline(-1.95, color=theme.AMBER, linewidth=1.2,
                    linestyle="--", label="5% CV")
        ax1.axvline(t_bar, color=theme.BLUE, linewidth=1.5,
                    linestyle="-", label=f"t̄ = {t_bar:.3f}")
        ax1.set_xlabel("ADF t-stat", color=theme.FG, fontsize=8)
        self._style_ax(ax1, "Individual ADF Statistics")
        ax1.legend(fontsize=7, facecolor=theme.WBG,
                   labelcolor=theme.FG, edgecolor=theme.GRAY)

        # Right: W_bar on N(0,1)
        x = np.linspace(-4, 4, 300)
        ax2.plot(x, _norm.pdf(x), color=theme.BLUE, linewidth=1.5, label="N(0,1)")
        ax2.fill_between(x, _norm.pdf(x), where=(x <= -1.645),
                         color=theme.RED, alpha=0.35, label="5% rejection region")
        ax2.axvline(W_bar, color=theme.AMBER, linewidth=2,
                    linestyle="--", label=f"W̄ = {W_bar:.3f}")
        ax2.set_xlabel("W-bar statistic", color=theme.FG, fontsize=8)
        self._style_ax(ax2, "IPS W̄ Distribution")
        ax2.legend(fontsize=7, facecolor=theme.WBG,
                   labelcolor=theme.FG, edgecolor=theme.GRAY)

        fig.suptitle(f"Im-Pesaran-Shin Panel Unit Root  —  {col}",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.ur_plot_frame, toolbar=True)

    def _plot_eg(self, col1, col2, df_c):
        """Levels + cointegration residuals for Engle-Granger."""
        self._clear_ur_plot()
        y = df_c[col1].values
        x = df_c[col2].values
        X = sm.add_constant(x)
        fit   = sm.OLS(y, X).fit()
        resid = fit.resid

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                        facecolor=theme.BG)
        fig.subplots_adjust(hspace=0.48, left=0.11, right=0.97,
                            top=0.90, bottom=0.10)

        ax1.plot(y, color=theme.BLUE, linewidth=1.2, label=col1)
        ax1.plot(fit.fittedvalues, color=theme.TEAL, linewidth=1.0,
                 linestyle="--", label=f"fitted")
        ax1.legend(fontsize=7, facecolor=theme.WBG,
                   labelcolor=theme.FG, edgecolor=theme.GRAY)
        self._style_ax(ax1, f"Levels: {col1} vs {col2}")

        ax2.plot(resid, color=theme.AMBER, linewidth=1.2)
        ax2.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        ax2.fill_between(range(len(resid)), resid, 0,
                         where=(resid >= 0),
                         color=theme.TEAL, alpha=0.28)
        ax2.fill_between(range(len(resid)), resid, 0,
                         where=(resid < 0),
                         color=theme.RED, alpha=0.28)
        self._style_ax(ax2, "Cointegration Residuals (ECM)")

        fig.suptitle("Engle-Granger Cointegration", color=theme.FG, fontsize=10)
        embed_figure(fig, self.ur_plot_frame, toolbar=True)

    def _plot_johansen(self, cols, res):
        """Trace stats vs critical values + eigenvalue bar chart."""
        self._clear_ur_plot()
        n      = len(cols)
        ev     = res.eig
        ts     = res.lr1
        cvt5   = res.cvt[:, 1]   # 5% critical values
        mx     = res.lr2
        cvm5   = res.cvm[:, 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4),
                                        facecolor=theme.BG)
        fig.subplots_adjust(wspace=0.38, left=0.09, right=0.97,
                            top=0.88, bottom=0.12)
        x = np.arange(n)

        # Trace vs 5% CV
        ax1.bar(x - 0.18, ts,    width=0.34, color=theme.BLUE,  label="Trace stat")
        ax1.bar(x + 0.18, cvt5,  width=0.34, color=theme.AMBER,
                alpha=0.80, label="5% CV")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"r≤{i}" for i in range(n)],
                             color=theme.FG, fontsize=8)
        ax1.legend(fontsize=8, facecolor=theme.WBG,
                   labelcolor=theme.FG, edgecolor=theme.GRAY)
        self._style_ax(ax1, "Trace Test")

        # Eigenvalues
        ax2.bar(x, ev, color=theme.TEAL, width=0.55)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"λ{i+1}" for i in range(n)],
                             color=theme.FG, fontsize=8)
        self._style_ax(ax2, "Eigenvalues")

        title_vars = ", ".join(cols[:3]) + ("…" if len(cols) > 3 else "")
        fig.suptitle(f"Johansen Cointegration  —  {title_vars}",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.ur_plot_frame, toolbar=True)

    def _plot_coint_heatmap(self, num_cols, rows):
        """Green/red heatmap of Johansen pairwise cointegration."""
        self._clear_ur_plot()
        n   = len(num_cols)
        mat = np.full((n, n), np.nan)
        idx = {c: i for i, c in enumerate(num_cols)}
        for c1, c2, tr, r in rows:
            i, j = idx[c1], idx[c2]
            mat[i, j] = mat[j, i] = float(r)

        fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(4, n * 0.65)),
                               facecolor=theme.BG)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(num_cols, rotation=45, ha="right",
                           fontsize=7, color=theme.FG)
        ax.set_yticklabels(num_cols, fontsize=7, color=theme.FG)
        for (i, j), val in np.ndenumerate(mat):
            if not np.isnan(val) and i != j:
                ax.text(j, i, "✓" if val else "–",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if val else theme.FG)
        self._style_ax(ax, "Johansen Cointegration Pairs (5% level)")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        fig.tight_layout()
        embed_figure(fig, self.ur_plot_frame, toolbar=True)

    # ── CIPS (Pesaran 2007) ───────────────────────────────────────────────
    def _cips_args(self):
        try:
            lags = max(0, min(int(self.ur_maxlag_var.get()), 3))
        except ValueError:
            lags = 1
        return lags, self.ur_trend.get() == "Constant + Trend"

    def _cips_header(self, lags, trend, r):
        return (f"  CADF lags = {lags}"
                f"{' (capped at 3 for short T)' if lags == 3 else ''};  "
                f"deterministics: {'constant + trend' if trend else 'constant'}"
                f";\n  balanced panel N = {r.N}, T = {r.T}; critical values "
                f"simulated (500 panels\n  with a common factor) for this "
                f"N and T.\n")

    def _run_cips(self):
        if not self._check_data():
            return
        col = self.ur_var.get()
        lags, trend = self._cips_args()
        df = self.df.copy()

        def _work():
            try:
                r = panel_tests.cips(df, col, lags, trend)
                verdict = ("REJECT H0 → stationary (in at least a "
                           "significant share of countries)" if r.reject
                           else "cannot reject H0 → unit root")
                out = (f"\n{'═'*60}\n Pesaran (2007) CIPS panel unit-root "
                       f"test — {col}\n{'═'*60}\n"
                       + self._cips_header(lags, trend, r) +
                       f"\n  CIPS = {r.stat:.3f}    p = {r.p_value:.3f}\n"
                       f"  critical: 1% {r.crit[0.01]:.3f}   5% "
                       f"{r.crit[0.05]:.3f}   10% {r.crit[0.10]:.3f}\n\n"
                       f"  H0: unit root in every country (robust to common "
                       f"shocks)\n  Conclusion: {verdict}\n")
                ind = r.individual.sort_values()
                out += "\n  Individual CADF t-statistics:\n" + "".join(
                    f"    {c:<16}{t:>8.3f}\n" for c, t in ind.items())

                def show():
                    clear_txt(self.ur_txt)
                    write(self.ur_txt, out)
                    self._plot_cips(r)
                self.root.after(0, show)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "CIPS Error", str(exc)))
        threading.Thread(target=_work, daemon=True).start()

    def _plot_cips(self, r):
        for w in self.ur_plot_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(9, 5.5), facecolor=theme.BG)
        ax = fig.add_subplot(111)
        ind = r.individual.sort_values()
        ax.barh(ind.index, ind.values, color=theme.BLUE, alpha=0.8)
        ax.axvline(r.stat, color=theme.TEAL, lw=2, label=f"CIPS {r.stat:.2f}")
        ax.axvline(r.crit[0.05], color=theme.RED, ls="--",
                   label=f"5 % critical {r.crit[0.05]:.2f}")
        ax.set_xlabel("CADF t-statistic")
        ax.set_title(f"CIPS — {r.variable}", color=theme.FG)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8)
        fig.tight_layout()
        embed_figure(fig, self.ur_plot_frame)

    def _run_cips_all(self):
        if not self._check_data():
            return
        lags, trend = self._cips_args()
        df = self.df.copy()
        cols = [c for c in df.select_dtypes("number").columns if c != "Year"]

        def _work():
            rows = []
            for c in cols:
                try:
                    r = panel_tests.cips(df, c, lags, trend, n_sim=300)
                    rows.append((c, r.stat, r.crit[0.05], r.p_value,
                                 r.N, "I(0)" if r.reject else "I(1)?"))
                except Exception as exc:
                    rows.append((c, np.nan, np.nan, np.nan, 0,
                                 f"skipped: {str(exc)[:30]}"))
            out = [f"\n{'═'*72}\n CIPS — all variables  (lags = {lags}, "
                   f"{'trend' if trend else 'constant'})\n{'═'*72}\n",
                   f"  {'Variable':<28}{'CIPS':>8}{'5% cv':>8}{'p':>8}"
                   f"{'N':>4}  verdict\n"]
            for c, st, cv, p, n, v in rows:
                out.append(f"  {c[:27]:<28}{st:>8.3f}{cv:>8.3f}{p:>8.3f}"
                           f"{n:>4}  {v}\n")
            text = "".join(out)
            self.root.after(0, lambda: (clear_txt(self.ur_txt),
                                        write(self.ur_txt, text)))
        threading.Thread(target=_work, daemon=True).start()

