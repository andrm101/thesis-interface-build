"""
tabs/tab_advanced.py  —  Advanced Econometrics tab.

Methods
───────
  · Quantile Regression           statsmodels QuantReg  (pooled, τ-path)
  · Mean Group (MG) Estimator     Pesaran & Smith (1995) — per-entity OLS avg.
  · Driscoll-Kraay Standard Errors Hoechle (2007) panel HAC sandwich
  · Local Projections IRF          Jordà (2005) — horizon-by-horizon OLS
  · Threshold Regression           Hansen (1999) — SSR grid search + F-test
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
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats as scipy_stats

import theme
from helpers import make_text, write, clear_txt, embed_figure


# ── Driscoll-Kraay covariance (Hoechle 2007) ─────────────────────────────────
def _dk_vcov(X, resid, year_idx, bandwidth=None):
    """
    Return the Driscoll-Kraay (1998) sandwich covariance matrix.

    Parameters
    ----------
    X         : (n_obs, k)  regressor matrix (with constant)
    resid     : (n_obs,)    OLS residuals
    year_idx  : (n_obs,)    integer year label for each observation
    bandwidth : int | None  Bartlett kernel width; default floor(4·(T/100)^{2/9})
    """
    years = np.sort(np.unique(year_idx))
    T     = len(years)
    k     = X.shape[1]
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    m = bandwidth

    # Cross-sectional average scores h_t  (T × k)
    H = np.zeros((T, k))
    for idx, yr in enumerate(years):
        mask       = year_idx == yr
        N_t        = mask.sum()
        H[idx]     = (X[mask].T @ resid[mask]) / N_t

    # Newey-West long-run covariance of {h_t} with Bartlett kernel
    S = H.T @ H                         # Γ_0
    for lag in range(1, m + 1):
        w       = 1.0 - lag / (m + 1)  # Bartlett weight
        gamma_l = H[lag:].T @ H[:-lag]
        S      += w * (gamma_l + gamma_l.T)
    S /= T

    # Sandwich estimator
    XTX_inv = np.linalg.pinv(X.T @ X)
    return T * XTX_inv @ S @ XTX_inv


# ── Helper ────────────────────────────────────────────────────────────────────
def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


class AdvancedTabMixin:

    # ── Tab builder ───────────────────────────────────────────────────────────
    def _tab_advanced(self, parent):
        # ── Row 1: core variable selection ────────────────────────────────────
        r1 = ttk.Frame(parent)
        r1.pack(fill=tk.X, padx=12, pady=(8, 2))

        # Dep var
        ttk.Label(r1, text="Dep. variable:").pack(side=tk.LEFT)
        self.adv_dep = ttk.Combobox(r1, width=20, state="readonly")
        self.adv_dep.pack(side=tk.LEFT, padx=(4, 12))

        # Regressors listbox
        rf = ttk.LabelFrame(r1, text="Regressors (select ≥ 1)")
        rf.pack(side=tk.LEFT, padx=(0, 12))
        self.adv_reg_lb = tk.Listbox(
            rf, selectmode=tk.MULTIPLE, height=4, width=26,
            bg=theme.WBG, fg=theme.FG,
            selectbackground=theme.BLUE,
            selectforeground=theme.WFG,
            exportselection=False)
        self.adv_reg_lb.pack(side=tk.LEFT)
        sb_r = ttk.Scrollbar(rf, orient=tk.VERTICAL,
                             command=self.adv_reg_lb.yview)
        sb_r.pack(side=tk.LEFT, fill=tk.Y)
        self.adv_reg_lb.configure(yscrollcommand=sb_r.set)

        # Right-side controls
        cf = ttk.Frame(r1)
        cf.pack(side=tk.LEFT, fill=tk.Y)

        def _lbl(text, row, col=0):
            ttk.Label(cf, text=text).grid(
                row=row, column=col, sticky=tk.W, padx=(0, 4), pady=2)

        _lbl("Threshold / LP var:", 0)
        self.adv_qvar = ttk.Combobox(cf, width=20, state="readonly")
        self.adv_qvar.grid(row=0, column=1, padx=4, pady=2)

        _lbl("Quantiles (space-sep.):", 1)
        self.adv_quantiles_var = tk.StringVar(value="0.10 0.25 0.50 0.75 0.90")
        ttk.Entry(cf, textvariable=self.adv_quantiles_var, width=26).grid(
            row=1, column=1, padx=4, pady=2)

        _lbl("LP / Threshold lags:", 2)
        self.adv_lags_var = tk.StringVar(value="2")
        sp_lag = tk.Spinbox(cf, from_=0, to=10, width=5,
                            textvariable=self.adv_lags_var,
                            bg=theme.WBG, fg=theme.FG,
                            insertbackground=theme.FG,
                            buttonbackground=theme.WBG)
        sp_lag.grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)
        self._tk_spinboxes.append(sp_lag)

        _lbl("LP horizons:", 3)
        self.adv_horizons_var = tk.StringVar(value="8")
        sp_hor = tk.Spinbox(cf, from_=2, to=30, width=5,
                            textvariable=self.adv_horizons_var,
                            bg=theme.WBG, fg=theme.FG,
                            insertbackground=theme.FG,
                            buttonbackground=theme.WBG)
        sp_hor.grid(row=3, column=1, sticky=tk.W, padx=4, pady=2)
        self._tk_spinboxes.append(sp_hor)

        _lbl("Country FE:", 4)
        self.adv_fe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, variable=self.adv_fe_var).grid(
            row=4, column=1, sticky=tk.W, padx=4, pady=2)

        # ── Button row 1 ──────────────────────────────────────────────────────
        r2 = ttk.Frame(parent)
        r2.pack(fill=tk.X, padx=12, pady=(6, 2))
        for label, cmd in [
            ("Quantile Regression",  self._run_quantile_reg),
            ("Mean Group Estimator", self._run_mg),
            ("Driscoll-Kraay SE",    self._run_dk_se),
        ]:
            ttk.Button(r2, text=label, command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Button row 2 ──────────────────────────────────────────────────────
        r3 = ttk.Frame(parent)
        r3.pack(fill=tk.X, padx=12, pady=(2, 6))
        adv_hdr = ttk.Label(r3, text="── Nonlinear / Time ──",
                            foreground=theme.TEAL)
        adv_hdr.pack(side=tk.LEFT, padx=(0, 10))
        self._special_labels.append((adv_hdr, "TEAL"))
        for label, cmd in [
            ("Local Projections IRF", self._run_lp_irf),
            ("Threshold Regression",  self._run_threshold),
        ]:
            ttk.Button(r3, text=label, command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Paned results ─────────────────────────────────────────────────────
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.adv_txt        = make_text(left, height=30, copy_root=self.root)
        self.adv_plot_frame = right

        self._refresh_adv_vars()

    # ── Refresh ───────────────────────────────────────────────────────────────
    def _refresh_adv_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.adv_dep["values"] = num_cols
        if num_cols:
            self.adv_dep.current(0)
        self.adv_reg_lb.delete(0, tk.END)
        for c in num_cols:
            self.adv_reg_lb.insert(tk.END, c)
        self.adv_qvar["values"] = num_cols
        if num_cols:
            self.adv_qvar.current(0)

    # ── Data helpers ──────────────────────────────────────────────────────────
    def _adv_get_xy(self, add_fe=True):
        """
        Return (y, X, year_idx, entity_idx, col_names) for pooled OLS.
        If add_fe is True and Country FE is checked, entity dummies are appended.
        """
        dep  = self.adv_dep.get()
        sel  = self.adv_reg_lb.curselection()
        regs = [self.adv_reg_lb.get(i) for i in sel]
        if not regs:
            raise ValueError("Select at least one regressor.")

        df    = self.df[[dep] + regs + ["Country", "Year"]].dropna()
        y     = df[dep].values
        X_raw = sm.add_constant(df[regs].values, has_constant="add")
        names = ["const"] + regs

        if add_fe and self.adv_fe_var.get():
            dummies = pd.get_dummies(df["Country"], drop_first=True).values
            X_raw = np.hstack([X_raw, dummies])
            names += [f"FE:{c}" for c in
                      pd.get_dummies(df["Country"],
                                     drop_first=True).columns.tolist()]

        year_idx   = df["Year"].values
        entity_idx = df["Country"].values
        return y, X_raw, year_idx, entity_idx, names, regs, df

    # ── Plot utility ──────────────────────────────────────────────────────────
    def _clear_adv_plot(self):
        for w in self.adv_plot_frame.winfo_children():
            w.destroy()

    def _style_ax(self, ax, title=None):
        ax.set_facecolor(theme.WBG)
        ax.tick_params(colors=theme.FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(theme.GRAY)
        if title:
            ax.set_title(title, color=theme.FG, fontsize=9)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. QUANTILE REGRESSION
    # ═════════════════════════════════════════════════════════════════════════
    def _run_quantile_reg(self):
        if not self._check_data():
            return

        def _work():
            try:
                dep  = self.adv_dep.get()
                sel  = self.adv_reg_lb.curselection()
                regs = [self.adv_reg_lb.get(i) for i in sel]
                if not regs:
                    raise ValueError("Select at least one regressor.")

                raw_q = self.adv_quantiles_var.get().split()
                taus  = [float(q) for q in raw_q if 0 < float(q) < 1]
                if not taus:
                    raise ValueError("Enter valid quantiles between 0 and 1.")

                df   = self.df[[dep] + regs].dropna()
                y    = df[dep].values
                X    = sm.add_constant(df[regs].values, has_constant="add")
                # OLS baseline
                ols_res   = sm.OLS(y, X).fit()
                ols_coefs = ols_res.params[1:]    # skip const
                ols_ci    = ols_res.conf_int()[1:]

                results_by_tau = {}
                for tau in taus:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        qr = QuantReg(y, X).fit(q=tau)
                    results_by_tau[tau] = qr

                # ── Text output ──────────────────────────────────────────────
                lines = [
                    f"\n{'═'*62}",
                    f" Quantile Regression  —  {dep}",
                    f"{'═'*62}",
                    f"  Regressors : {', '.join(regs)}",
                    f"  Obs        : {len(y)}",
                    f"",
                    f"  {'Variable':<22} " +
                    "".join(f" {'τ='+str(t):>9}" for t in taus) +
                    f"  {'OLS':>9}",
                    f"  {'─'*56}",
                ]
                for idx, reg in enumerate(regs):
                    coef_row = f"  {reg:<22}"
                    for tau in taus:
                        coef = results_by_tau[tau].params[idx + 1]
                        s    = _stars(results_by_tau[tau].pvalues[idx + 1])
                        coef_row += f" {coef:>7.4f}{s:>2}"
                    coef_row += f" {ols_coefs[idx]:>9.4f}"
                    lines.append(coef_row)
                    # p-value row
                    pval_row = f"  {'':22}"
                    for tau in taus:
                        p = results_by_tau[tau].pvalues[idx + 1]
                        pval_row += f" [{p:>7.4f}]"
                    pval_row += f" [{ols_res.pvalues[idx+1]:>7.4f}]"
                    lines.append(pval_row)
                    lines.append("")

                lines.append("  p-values in brackets; *** p<0.01  ** p<0.05  * p<0.10")
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.adv_txt),
                                            write(self.adv_txt, out)))
                self.root.after(
                    0, lambda: self._plot_quantile_path(
                        regs, taus, results_by_tau,
                        ols_coefs, ols_ci))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Quantile Reg Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _plot_quantile_path(self, regs, taus, results, ols_coefs, ols_ci):
        self._clear_adv_plot()
        k   = len(regs)
        fig, axes = plt.subplots(
            k, 1, figsize=(7, max(3, 2.5 * k)),
            facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(hspace=0.55, left=0.14, right=0.97,
                            top=0.93, bottom=0.06)

        for i, reg in enumerate(regs):
            ax      = axes[i, 0]
            coefs   = [results[t].params[i + 1] for t in taus]
            ci_lo   = [results[t].conf_int()[i + 1, 0] for t in taus]
            ci_hi   = [results[t].conf_int()[i + 1, 1] for t in taus]

            ax.plot(taus, coefs, color=theme.BLUE, linewidth=1.8,
                    marker="o", markersize=4, label="QR coeff.")
            ax.fill_between(taus, ci_lo, ci_hi,
                            color=theme.BLUE, alpha=0.18, label="95% CI")
            # OLS baseline
            ax.axhline(ols_coefs[i], color=theme.AMBER,
                       linewidth=1.4, linestyle="--", label="OLS")
            ax.fill_between([taus[0], taus[-1]],
                            ols_ci[i][0], ols_ci[i][1],
                            color=theme.AMBER, alpha=0.12)
            ax.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle=":")
            ax.set_xlabel("Quantile τ", color=theme.FG, fontsize=8)
            ax.legend(fontsize=7, facecolor=theme.WBG,
                      labelcolor=theme.FG, edgecolor=theme.GRAY)
            self._style_ax(ax, reg)

        fig.suptitle("Quantile Regression — Coefficient Paths",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.adv_plot_frame, toolbar=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. MEAN GROUP (MG) ESTIMATOR
    # ═════════════════════════════════════════════════════════════════════════
    def _run_mg(self):
        if not self._check_data():
            return

        def _work():
            try:
                dep  = self.adv_dep.get()
                sel  = self.adv_reg_lb.curselection()
                regs = [self.adv_reg_lb.get(i) for i in sel]
                if not regs:
                    raise ValueError("Select at least one regressor.")

                df       = self.df[[dep] + regs + ["Country"]].dropna()
                entities = df["Country"].unique()
                coef_store = {r: [] for r in regs}
                se_store   = {r: [] for r in regs}
                valid_ents = []

                for ent in entities:
                    sub = df[df["Country"] == ent]
                    if len(sub) < len(regs) + 2:
                        continue
                    yi   = sub[dep].values
                    Xi   = sm.add_constant(sub[regs].values, has_constant="add")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res_i = sm.OLS(yi, Xi).fit()
                    valid_ents.append(ent)
                    for j, r in enumerate(regs):
                        coef_store[r].append(res_i.params[j + 1])
                        se_store[r].append(res_i.bse[j + 1])

                N = len(valid_ents)
                if N < 2:
                    raise ValueError("Need ≥ 2 countries with sufficient obs.")

                # MG mean and variance (Pesaran & Smith 1995)
                mg_coefs = {}
                mg_ses   = {}
                for r in regs:
                    c_arr     = np.array(coef_store[r])
                    mg_mean   = c_arr.mean()
                    mg_var    = c_arr.var(ddof=1) / N   # variance of sample mean
                    mg_coefs[r] = mg_mean
                    mg_ses[r]   = np.sqrt(mg_var)

                # Pooled FE baseline for comparison
                X_fe = sm.add_constant(df[regs].values, has_constant="add")
                fe_dummies = pd.get_dummies(df["Country"], drop_first=True).values
                X_fe = np.hstack([X_fe, fe_dummies])
                fe_res = sm.OLS(df[dep].values, X_fe).fit()

                lines = [
                    f"\n{'═'*60}",
                    f" Mean Group (MG) Estimator  —  {dep}",
                    f"{'═'*60}",
                    f"  Method     : Pesaran & Smith (1995)",
                    f"  Entities   : {N} countries",
                    f"  Regressors : {', '.join(regs)}",
                    f"",
                    f"  {'Variable':<24} {'MG coeff':>10} {'MG SE':>10}"
                    f"  {'t-stat':>8}  Sig  {'FE coeff':>10}",
                    f"  {'─'*68}",
                ]
                for r in regs:
                    mg_c  = mg_coefs[r]
                    mg_se = mg_ses[r]
                    t_mg  = mg_c / mg_se if mg_se > 0 else np.nan
                    p_mg  = 2 * (1 - scipy_stats.t.cdf(abs(t_mg), df=N - 1))
                    # FE coeff
                    fe_idx = list(df.columns).index(r) if r in df.columns else -1
                    fe_c = fe_res.params[regs.index(r) + 1]
                    lines.append(
                        f"  {r:<24} {mg_c:>10.5f} {mg_se:>10.5f}"
                        f"  {t_mg:>8.3f}  {_stars(p_mg):<3}  {fe_c:>10.5f}")

                # Heterogeneity test (spread of individual coefficients)
                lines += [f"", f"  Per-country coefficients:"]
                for r in regs:
                    c_arr = np.array(coef_store[r])
                    lines.append(
                        f"  {r:<22}  min={c_arr.min():>8.4f}  "
                        f"max={c_arr.max():>8.4f}  "
                        f"sd={c_arr.std():>8.4f}")
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.adv_txt),
                                            write(self.adv_txt, out)))
                self.root.after(
                    0, lambda: self._plot_mg(
                        regs, coef_store, mg_coefs, mg_ses, valid_ents))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "MG Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _plot_mg(self, regs, coef_store, mg_coefs, mg_ses, entities):
        self._clear_adv_plot()
        k   = len(regs)
        fig, axes = plt.subplots(
            1, k, figsize=(max(5, 3.5 * k), 5),
            facecolor=theme.BG, squeeze=False)
        fig.subplots_adjust(wspace=0.40, left=0.10, right=0.97,
                            top=0.90, bottom=0.14)

        for i, reg in enumerate(regs):
            ax     = axes[0, i]
            c_arr  = np.array(coef_store[reg])
            y_pos  = np.arange(len(entities))
            colors = [theme.TEAL if c > 0 else theme.RED for c in c_arr]
            ax.barh(y_pos, c_arr, color=colors, height=0.6, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(entities, fontsize=6, color=theme.FG)
            # MG mean ± 2 SE
            ax.axvline(mg_coefs[reg], color=theme.BLUE, linewidth=2.0,
                       label=f"MG = {mg_coefs[reg]:.4f}")
            ax.axvline(mg_coefs[reg] + 2 * mg_ses[reg],
                       color=theme.BLUE, linewidth=0.8, linestyle="--")
            ax.axvline(mg_coefs[reg] - 2 * mg_ses[reg],
                       color=theme.BLUE, linewidth=0.8, linestyle="--")
            ax.axvline(0, color=theme.GRAY, linewidth=0.8, linestyle=":")
            ax.legend(fontsize=7, facecolor=theme.WBG,
                      labelcolor=theme.FG, edgecolor=theme.GRAY)
            self._style_ax(ax, reg)

        fig.suptitle("Mean Group Estimator — Per-Country Coefficients",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.adv_plot_frame, toolbar=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. DRISCOLL-KRAAY STANDARD ERRORS
    # ═════════════════════════════════════════════════════════════════════════
    def _run_dk_se(self):
        if not self._check_data():
            return

        def _work():
            try:
                y, X, year_idx, entity_idx, names, regs, df_used = \
                    self._adv_get_xy(add_fe=True)

                # OLS
                ols_res = sm.OLS(y, X).fit()
                resid   = ols_res.resid

                # Driscoll-Kraay VCV
                vcov_dk = _dk_vcov(X, resid, year_idx)
                dk_se   = np.sqrt(np.diag(vcov_dk))
                dk_t    = ols_res.params / dk_se
                df_resid = len(y) - X.shape[1]
                dk_p    = 2 * scipy_stats.t.sf(np.abs(dk_t), df=df_resid)

                # Extract only regressor rows (skip const + FE dummies)
                n_regs  = len(regs) + 1   # +1 for const
                lines   = [
                    f"\n{'═'*66}",
                    f" Driscoll-Kraay Standard Errors",
                    f"{'═'*66}",
                    f"  Method    : Hoechle (2007) panel HAC sandwich",
                    f"  Country FE: {'Yes' if self.adv_fe_var.get() else 'No'}",
                    f"  Obs       : {len(y)}",
                    f"  Bandwidth : floor(4·(T/100)^{{2/9}})  "
                    f"(T = {len(np.unique(year_idx))})",
                    f"",
                    f"  {'Variable':<24} {'Coeff':>10} {'OLS SE':>10}"
                    f" {'DK SE':>10} {'DK t':>8} {'DK p':>8}  Sig",
                    f"  {'─'*70}",
                ]
                for j in range(n_regs):
                    nm  = names[j]
                    if nm.startswith("FE:"):
                        continue
                    b   = ols_res.params[j]
                    se_ols = ols_res.bse[j]
                    se_dk  = dk_se[j]
                    t_dk   = dk_t[j]
                    p_dk   = dk_p[j]
                    lines.append(
                        f"  {nm:<24} {b:>10.5f} {se_ols:>10.5f}"
                        f" {se_dk:>10.5f} {t_dk:>8.3f} {p_dk:>8.4f}"
                        f"  {_stars(p_dk)}")

                lines += [
                    f"",
                    f"  R² (OLS)   : {ols_res.rsquared:.4f}",
                    f"  Adj R²     : {ols_res.rsquared_adj:.4f}",
                    f"  *** p<0.01  ** p<0.05  * p<0.10",
                ]
                out = "\n".join(lines) + "\n"
                # Store for Report tab capture
                self._adv_last_result = ols_res
                self.root.after(0, lambda: (clear_txt(self.adv_txt),
                                            write(self.adv_txt, out)))
                self.root.after(
                    0, lambda: self._plot_dk_comparison(
                        regs, ols_res, dk_se, dk_p, n_regs))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "DK Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _plot_dk_comparison(self, regs, ols_res, dk_se, dk_p, n_regs):
        self._clear_adv_plot()
        k        = len(regs)
        coefs    = ols_res.params[1:k + 1]
        se_ols   = ols_res.bse[1:k + 1]
        se_dk    = dk_se[1:k + 1]

        y_pos = np.arange(k)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3, k * 0.6 + 1.5)),
                                        facecolor=theme.BG)
        fig.subplots_adjust(wspace=0.38, left=0.18, right=0.97,
                            top=0.88, bottom=0.12)

        for ax, se, label, col in [
            (ax1, se_ols, "OLS SE",          theme.BLUE),
            (ax2, se_dk,  "Driscoll-Kraay SE", theme.AMBER),
        ]:
            ax.errorbar(coefs, y_pos,
                        xerr=1.96 * se,
                        fmt="o", color=col, ecolor=col,
                        capsize=4, markersize=5, linewidth=1.4)
            ax.axvline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(regs, fontsize=8, color=theme.FG)
            ax.set_xlabel("Coefficient ± 1.96·SE", color=theme.FG, fontsize=8)
            self._style_ax(ax, label)

        fig.suptitle("OLS vs Driscoll-Kraay Standard Errors",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.adv_plot_frame, toolbar=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. LOCAL PROJECTIONS IRF  (Jordà 2005)
    # ═════════════════════════════════════════════════════════════════════════
    def _run_lp_irf(self):
        if not self._check_data():
            return

        def _work():
            try:
                dep    = self.adv_dep.get()
                shock  = self.adv_qvar.get()    # shock variable
                sel    = self.adv_reg_lb.curselection()
                ctrls  = [self.adv_reg_lb.get(i) for i in sel
                          if self.adv_reg_lb.get(i) != shock]
                lags   = int(self.adv_lags_var.get())
                H      = int(self.adv_horizons_var.get())
                use_fe = self.adv_fe_var.get()

                needed = [dep, shock] + ctrls + ["Country", "Year"]
                df_lp  = self.df[list(dict.fromkeys(needed))].copy()

                betas, lo90, hi90, lo68, hi68 = [], [], [], [], []

                for h in range(H + 1):
                    rows = []
                    for _, grp in df_lp.groupby("Country"):
                        g = grp.sort_values("Year").copy()
                        # Forward y (level change from t to t+h)
                        g["_y_fwd"] = g[dep].shift(-h) - g[dep].shift(1)
                        # Lag controls
                        for lag in range(1, lags + 1):
                            g[f"_dep_lag{lag}"] = g[dep].shift(lag)
                            g[f"_shk_lag{lag}"] = g[shock].shift(lag)
                        g = g.dropna()
                        if len(g) < lags + 3:
                            continue
                        rows.append(g)

                    if not rows:
                        betas.append(np.nan)
                        for lst in (lo90, hi90, lo68, hi68):
                            lst.append(np.nan)
                        continue

                    df_h   = pd.concat(rows, ignore_index=True)
                    lag_cols = ([f"_dep_lag{l}" for l in range(1, lags + 1)]
                                + [f"_shk_lag{l}" for l in range(1, lags + 1)])
                    rhs_cols = [shock] + ctrls + lag_cols
                    Xh = sm.add_constant(df_h[rhs_cols].values,
                                         has_constant="add")
                    if use_fe:
                        fe_dum = pd.get_dummies(
                            df_h["Country"], drop_first=True).values
                        Xh = np.hstack([Xh, fe_dum])
                    yh = df_h["_y_fwd"].values

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res_h = sm.OLS(yh, Xh).fit()

                    # Use DK SE for inference
                    resid_h   = res_h.resid
                    year_h    = df_h["Year"].values
                    vcov_dk_h = _dk_vcov(Xh, resid_h, year_h)
                    se_h      = np.sqrt(vcov_dk_h[1, 1])   # shock is col 1

                    beta_h = res_h.params[1]
                    betas.append(beta_h)
                    lo90.append(beta_h - 1.645 * se_h)
                    hi90.append(beta_h + 1.645 * se_h)
                    lo68.append(beta_h - 1.000 * se_h)
                    hi68.append(beta_h + 1.000 * se_h)

                lines = [
                    f"\n{'═'*60}",
                    f" Local Projections IRF  (Jordà, 2005)",
                    f"{'═'*60}",
                    f"  Response  : {dep}",
                    f"  Shock     : {shock}",
                    f"  Controls  : {', '.join(ctrls) if ctrls else 'none'}",
                    f"  Lags (p)  : {lags}",
                    f"  Country FE: {'Yes' if use_fe else 'No'}",
                    f"  SE        : Driscoll-Kraay HAC",
                    f"",
                    f"  {'h':>4} {'β(h)':>10} {'Lo 90%':>10} {'Hi 90%':>10}",
                    f"  {'─'*40}",
                ]
                for h in range(H + 1):
                    b  = betas[h]
                    lo = lo90[h]
                    hi = hi90[h]
                    if np.isnan(b):
                        lines.append(f"  {h:>4}  (insufficient obs.)")
                    else:
                        lines.append(
                            f"  {h:>4} {b:>10.5f} {lo:>10.5f} {hi:>10.5f}")
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.adv_txt),
                                            write(self.adv_txt, out)))
                self.root.after(
                    0, lambda: self._plot_lp_irf(
                        dep, shock, H, betas, lo90, hi90, lo68, hi68))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "LP-IRF Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _plot_lp_irf(self, dep, shock, H, betas, lo90, hi90, lo68, hi68):
        self._clear_adv_plot()
        horizons = np.arange(H + 1)
        b  = np.array(betas, dtype=float)
        l9 = np.array(lo90, dtype=float)
        h9 = np.array(hi90, dtype=float)
        l6 = np.array(lo68, dtype=float)
        h6 = np.array(hi68, dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4), facecolor=theme.BG)
        self._style_ax(ax, f"LP-IRF: {dep} response to {shock} shock")
        ax.fill_between(horizons, l9, h9,
                        color=theme.BLUE, alpha=0.18, label="90% CI (DK)")
        ax.fill_between(horizons, l6, h6,
                        color=theme.BLUE, alpha=0.32, label="68% CI (DK)")
        ax.plot(horizons, b, color=theme.BLUE, linewidth=2.0,
                marker="o", markersize=4, label="LP-IRF")
        ax.axhline(0, color=theme.GRAY, linewidth=0.9, linestyle="--")
        ax.set_xlabel("Horizon h", color=theme.FG, fontsize=9)
        ax.set_ylabel("Cumulative response", color=theme.FG, fontsize=9)
        ax.legend(fontsize=8, facecolor=theme.WBG,
                  labelcolor=theme.FG, edgecolor=theme.GRAY)
        fig.tight_layout()
        embed_figure(fig, self.adv_plot_frame, toolbar=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 5. THRESHOLD REGRESSION  (Hansen 1999 — SSR grid search)
    # ═════════════════════════════════════════════════════════════════════════
    def _run_threshold(self):
        if not self._check_data():
            return

        def _work():
            try:
                dep     = self.adv_dep.get()
                q_var   = self.adv_qvar.get()    # threshold variable
                sel     = self.adv_reg_lb.curselection()
                regs    = [self.adv_reg_lb.get(i) for i in sel]
                if not regs:
                    raise ValueError("Select at least one regressor.")

                all_cols = list(dict.fromkeys(
                    [dep, q_var] + regs + ["Country"]))
                df_t = self.df[all_cols].dropna().copy()
                y    = df_t[dep].values
                Q    = df_t[q_var].values
                X    = sm.add_constant(df_t[regs].values, has_constant="add")
                if self.adv_fe_var.get():
                    fe_d = pd.get_dummies(
                        df_t["Country"], drop_first=True).values
                    X = np.hstack([X, fe_d])

                n = len(y)
                # Candidate thresholds: 10th–90th percentile, 100 grid points
                q_lo, q_hi = np.percentile(Q, 10), np.percentile(Q, 90)
                grid       = np.linspace(q_lo, q_hi, 100)
                ssr_grid   = np.full(len(grid), np.nan)

                k = X.shape[1]
                for g_idx, gamma in enumerate(grid):
                    d      = (Q > gamma).astype(float)
                    # Regime interaction: X_low = X*(1-d), X_high = X*d
                    X_low  = X * (1 - d)[:, None]
                    X_high = X * d[:, None]
                    Xg     = np.hstack([X_low, X_high])
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res_g  = sm.OLS(y, Xg).fit()
                        ssr_grid[g_idx] = res_g.ssr
                    except Exception:
                        pass

                # Optimal threshold
                opt_idx   = np.nanargmin(ssr_grid)
                gamma_opt = grid[opt_idx]
                d_opt     = (Q > gamma_opt).astype(float)
                X_low     = X * (1 - d_opt)[:, None]
                X_high    = X * d_opt[:, None]
                Xg_opt    = np.hstack([X_low, X_high])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_opt = sm.OLS(y, Xg_opt).fit()

                # Null (no threshold): pooled OLS
                res_null  = sm.OLS(y, X).fit()

                # F-test for threshold (Hansen 1999 supremum F)
                ssr_null  = res_null.ssr
                ssr_alt   = res_opt.ssr
                df_rest   = k            # number of restrictions
                F_thresh  = ((ssr_null - ssr_alt) / df_rest) / (ssr_alt / res_opt.df_resid)
                p_F       = scipy_stats.f.sf(F_thresh, df_rest, res_opt.df_resid)

                n_low  = int((1 - d_opt).sum())
                n_high = int(d_opt.sum())

                lines = [
                    f"\n{'═'*62}",
                    f" Threshold Regression  —  {dep}",
                    f"{'═'*62}",
                    f"  Threshold variable  : {q_var}",
                    f"  Regressors          : {', '.join(regs)}",
                    f"  Country FE          : {'Yes' if self.adv_fe_var.get() else 'No'}",
                    f"  Grid search range   : [{q_lo:.4f}, {q_hi:.4f}]  (100 pts)",
                    f"",
                    f"  Optimal threshold γ*: {gamma_opt:.6f}",
                    f"  Obs. below threshold: {n_low}",
                    f"  Obs. above threshold: {n_high}",
                    f"",
                    f"  F-test for threshold (H₀: no threshold)",
                    f"  F-statistic         : {F_thresh:>10.4f}",
                    f"  p-value             : {p_F:>10.4f}  {_stars(p_F)}",
                    f"  (Note: asymptotic F; for exact inference use bootstrap)",
                    f"",
                ]
                n_base = k   # base regressors count per regime
                reg_names = (["const"] + regs +
                             [f"FE:{c}" for c in
                              pd.get_dummies(df_t["Country"],
                                             drop_first=True).columns.tolist()]
                             if self.adv_fe_var.get()
                             else ["const"] + regs)
                lines += [f"  {'Variable':<24} {'Low regime':>12} {'High regime':>12}",
                          f"  {'─'*52}"]
                params = res_opt.params
                for j in range(k):
                    nm = reg_names[j] if j < len(reg_names) else f"var{j}"
                    if nm.startswith("FE:"):
                        continue
                    b_lo = params[j]
                    b_hi = params[j + k]
                    lines.append(f"  {nm:<24} {b_lo:>12.5f} {b_hi:>12.5f}")

                lines += [f"", f"  Model fit:",
                          f"  R² (threshold)  : {res_opt.rsquared:.4f}",
                          f"  R² (pooled OLS) : {res_null.rsquared:.4f}",
                          f"  SSR reduction   : {(ssr_null - ssr_alt)/ssr_null*100:.2f}%"]
                out = "\n".join(lines) + "\n"
                self.root.after(0, lambda: (clear_txt(self.adv_txt),
                                            write(self.adv_txt, out)))
                self.root.after(
                    0, lambda: self._plot_threshold(
                        df_t, dep, q_var, regs,
                        gamma_opt, grid, ssr_grid, d_opt))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Threshold Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _plot_threshold(self, df_t, dep, q_var, regs, gamma_opt,
                        grid, ssr_grid, d_opt):
        self._clear_adv_plot()
        Q = df_t[q_var].values
        y = df_t[dep].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                        facecolor=theme.BG)
        fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97,
                            top=0.88, bottom=0.12)

        # Left: scatter coloured by regime
        low_mask  = d_opt == 0
        high_mask = d_opt == 1
        x_reg     = df_t[regs[0]].values
        ax1.scatter(x_reg[low_mask],  y[low_mask],
                    color=theme.BLUE,  s=18, alpha=0.65, label="Low regime")
        ax1.scatter(x_reg[high_mask], y[high_mask],
                    color=theme.AMBER, s=18, alpha=0.65, label="High regime")
        ax1.set_xlabel(regs[0], color=theme.FG, fontsize=8)
        ax1.set_ylabel(dep, color=theme.FG, fontsize=8)
        ax1.legend(fontsize=7, facecolor=theme.WBG,
                   labelcolor=theme.FG, edgecolor=theme.GRAY)
        self._style_ax(ax1, f"Regimes split by {q_var} = {gamma_opt:.4f}")

        # Right: SSR vs threshold grid
        ax2.plot(grid, ssr_grid, color=theme.BLUE, linewidth=1.4)
        ax2.axvline(gamma_opt, color=theme.AMBER, linewidth=2.0,
                    linestyle="--", label=f"γ* = {gamma_opt:.4f}")
        ax2.set_xlabel(f"Threshold γ  ({q_var})", color=theme.FG, fontsize=8)
        ax2.set_ylabel("SSR", color=theme.FG, fontsize=8)
        ax2.legend(fontsize=8, facecolor=theme.WBG,
                   labelcolor=theme.FG, edgecolor=theme.GRAY)
        self._style_ax(ax2, "Grid Search — SSR vs Threshold")

        fig.suptitle("Threshold Regression  (Hansen 1999)",
                     color=theme.FG, fontsize=10)
        embed_figure(fig, self.adv_plot_frame, toolbar=True)
