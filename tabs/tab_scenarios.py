"""
tabs/tab_scenarios.py  —  Growth scenario simulator and J-Curve policy shock.
Mixin: ScenariosTabMixin
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import theme
from helpers import make_text, write, clear_txt, embed_figure


class ScenariosTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — GROWTH SCENARIOS
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_scenarios(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Growth Scenario Simulator",
                              padding=10)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        _hint = ttk.Label(
            ctrl,
            text="Uses the best-trained ML model. "
                 "Train models in the ML tab first.",
            foreground=theme.AMBER,
        )
        _hint.pack(anchor=tk.W, pady=2)
        self._special_labels.append((_hint, "AMBER"))

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Horizon (years):").pack(side=tk.LEFT)
        self.sc_years = tk.IntVar(value=5)
        _spsc = tk.Spinbox(row0, from_=1, to=20, width=4,
                           textvariable=self.sc_years,
                           bg=theme.WBG, fg=theme.FG,
                           insertbackground=theme.FG)
        _spsc.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_spsc)
        ttk.Label(row0, text="  Country:").pack(side=tk.LEFT, padx=10)
        self.sc_country = ttk.Combobox(row0, width=22, state="readonly")
        self.sc_country.pack(side=tk.LEFT, padx=4)

        # Variable sliders
        sl_frm = ttk.LabelFrame(ctrl,
                                text="% Change per Year for Key Variables",
                                padding=8)
        sl_frm.pack(fill=tk.X, pady=4)
        self.sc_slider_vars = [
            "PIB towards research", "Labor in research",
            "Human Capital Proxy",  "Savings Percentage",
        ]
        self.sc_sliders = {}
        for var in self.sc_slider_vars:
            row = ttk.Frame(sl_frm); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=var, width=28).pack(side=tk.LEFT)
            sv  = tk.DoubleVar(value=0.0)
            lbl = ttk.Label(row, text="  +0.0%", width=8, foreground=theme.TEAL)
            ttk.Scale(row, from_=-20, to=20, variable=sv,
                      orient=tk.HORIZONTAL, length=260).pack(side=tk.LEFT, padx=6)
            lbl.pack(side=tk.LEFT)
            sv.trace_add("write",
                         lambda *a, s=sv, l=lbl: l.config(
                             text=f"{s.get():>+6.1f}%"))
            self.sc_sliders[var] = sv

        preset_row = ttk.Frame(ctrl); preset_row.pack(fill=tk.X, pady=3)
        ttk.Label(preset_row, text="Presets:").pack(side=tk.LEFT)
        for label, val in [("Optimistic (+5%)",  5),
                           ("Baseline (0%)",      0),
                           ("Pessimistic (-5%)", -5)]:
            ttk.Button(preset_row, text=label,
                       command=lambda v=val: self._set_preset(v)).pack(
                           side=tk.LEFT, padx=4)
        ttk.Button(preset_row, text="Run All 3 + Custom",
                   style="Accent.TButton",
                   command=self._run_scenarios).pack(side=tk.LEFT, padx=14)

        # J-Curve shock simulator
        jc_frm = ttk.LabelFrame(
            ctrl,
            text="J-Curve Policy Shock  "
                 "(initial productivity dip → long-run recovery)",
            padding=8,
        )
        jc_frm.pack(fill=tk.X, pady=6)
        jc_hint = ttk.Label(
            jc_frm,
            text="Models the absorption-capacity lag: R&D investment depresses "
                 "output before driving endogenous growth.",
            foreground=theme.TEAL,
        )
        jc_hint.pack(anchor=tk.W, pady=(0, 4))
        self._special_labels.append((jc_hint, "TEAL"))

        jc_row1 = ttk.Frame(jc_frm); jc_row1.pack(fill=tk.X, pady=2)
        ttk.Label(jc_row1,
                  text="Shock depth (% output loss at trough):").pack(
            side=tk.LEFT)
        self.jc_depth    = tk.DoubleVar(value=10.0)
        jc_depth_lbl     = ttk.Label(jc_row1, text=" −10.0%", width=8,
                                     foreground=theme.RED)
        ttk.Scale(jc_row1, from_=0, to=40, variable=self.jc_depth,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=6)
        jc_depth_lbl.pack(side=tk.LEFT)
        self._special_labels.append((jc_depth_lbl, "RED"))
        self.jc_depth.trace_add("write", lambda *a: jc_depth_lbl.config(
            text=f" −{self.jc_depth.get():.1f}%"))

        jc_row2 = ttk.Frame(jc_frm); jc_row2.pack(fill=tk.X, pady=2)
        ttk.Label(jc_row2, text="Trough year:").pack(side=tk.LEFT)
        self.jc_trough = tk.IntVar(value=3)
        _spjt = tk.Spinbox(jc_row2, from_=1, to=10, width=3,
                           textvariable=self.jc_trough,
                           bg=theme.WBG, fg=theme.FG,
                           insertbackground=theme.FG)
        _spjt.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_spjt)
        ttk.Label(jc_row2,
                  text="   Long-run gain (% above baseline):").pack(
            side=tk.LEFT, padx=10)
        self.jc_ltgain   = tk.DoubleVar(value=8.0)
        jc_ltgain_lbl    = ttk.Label(jc_row2, text=" +8.0%", width=8,
                                     foreground=theme.TEAL)
        ttk.Scale(jc_row2, from_=0, to=30, variable=self.jc_ltgain,
                  orient=tk.HORIZONTAL, length=160).pack(side=tk.LEFT, padx=6)
        jc_ltgain_lbl.pack(side=tk.LEFT)
        self._special_labels.append((jc_ltgain_lbl, "TEAL"))
        self.jc_ltgain.trace_add("write", lambda *a: jc_ltgain_lbl.config(
            text=f" +{self.jc_ltgain.get():.1f}%"))

        ttk.Button(jc_frm, text="Run J-Curve Simulation",
                   style="Accent.TButton",
                   command=self._run_jcurve).pack(anchor=tk.W, pady=4)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.sc_txt        = make_text(left, height=25, copy_root=self.root)
        self.sc_plot_frame = right

    def _refresh_sc_countries(self):
        if self.df is None:
            return
        countries = sorted(self.df["Country"].unique())
        self.sc_country["values"] = ["All Countries"] + countries
        self.sc_country.current(0)

    def _set_preset(self, val: float):
        for sv in self.sc_sliders.values():
            sv.set(float(val))

    # ── Multi-scenario run ────────────────────────────────────────────────────
    def _run_scenarios(self):
        if self.ml_model is None:
            messagebox.showwarning("No Model",
                "Train an ML model first in the ML Models tab.")
            return
        if not self._check_data():
            return

        horizon = self.sc_years.get()
        country = self.sc_country.get()
        custom  = {v: self.sc_sliders[v].get() / 100.0
                   for v in self.sc_slider_vars}

        df = self.df.copy()
        if country != "All Countries":
            df = df[df["Country"] == country]
        last_yr  = df["Year"].max()
        base_df  = df[df["Year"] == last_yr]
        target   = (self.ml_target.get()
                    if hasattr(self, "ml_target") and self.ml_target.get()
                    else "Y by L")

        scenarios = {
            "Optimistic (+5%)":  {v:  0.05 for v in self.sc_slider_vars},
            "Baseline (0%)":     {v:  0.00 for v in self.sc_slider_vars},
            "Pessimistic (-5%)": {v: -0.05 for v in self.sc_slider_vars},
            "Custom":             custom,
        }

        clear_txt(self.sc_txt)
        write(self.sc_txt,
              f"GROWTH SCENARIO ANALYSIS\n{'='*52}\n"
              f"Country    : {country}\n"
              f"Base year  : {last_yr}\n"
              f"Horizon    : {horizon} years\n"
              f"Target var : {target}\n\n")

        all_preds = {}
        base_val  = (base_df[target].mean()
                     if target in base_df.columns else None)

        for sc_name, sc_chg in scenarios.items():
            proj_rows = []
            for yr in range(1, horizon + 1):
                row_data = {}
                for f in self.ml_features:
                    base_f = f.replace("_lag1", "")
                    if base_f in sc_chg and base_f in base_df.columns:
                        row_data[f] = (base_df[base_f].mean() *
                                       (1 + sc_chg[base_f]) ** yr)
                    elif base_f in base_df.columns:
                        row_data[f] = base_df[base_f].mean()
                    else:
                        row_data[f] = 0.0
                proj_rows.append(row_data)

            X_proj = pd.DataFrame(proj_rows)[self.ml_features]
            X_sc   = self.ml_scaler.transform(X_proj)
            preds  = self.ml_model.predict(X_sc)
            all_preds[sc_name] = preds

            write(self.sc_txt, f"\n{sc_name}:\n")
            for i, p in enumerate(preds, 1):
                chg = ((p / base_val - 1) * 100) if base_val else 0
                write(self.sc_txt,
                      f"  {last_yr + i}: {target} = {p:>9.3f}"
                      f"  ({chg:>+6.2f}% vs base)\n")

        for w in self.sc_plot_frame.winfo_children():
            w.destroy()
        fig         = Figure(figsize=(9, 6), facecolor=theme.BG)
        ax          = fig.add_subplot(111)
        years_proj  = [last_yr + i for i in range(1, horizon + 1)]
        sc_pal      = {
            "Optimistic (+5%)":  theme.TEAL,
            "Baseline (0%)":     theme.BLUE,
            "Pessimistic (-5%)": theme.RED,
            "Custom":            theme.AMBER,
        }
        if base_val is not None:
            ax.axhline(base_val, color=theme.GRAY, linewidth=1.2,
                       linestyle=":",
                       label=f"Base ({last_yr}) = {base_val:.3f}")
        for sc_name, preds in all_preds.items():
            ax.plot(years_proj, preds, marker="o", markersize=5,
                    color=sc_pal.get(sc_name, theme.GRAY),
                    linewidth=2, label=sc_name, alpha=0.9)
        ax.set_xlabel("Year",   color=theme.FG)
        ax.set_ylabel(target,   color=theme.FG)
        ax.set_title(f"Growth Scenarios — {country}", color=theme.FG)
        ax.legend(facecolor=theme.WBG, labelcolor=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.sc_plot_frame)
        self.status_var.set(
            f"Scenarios complete — {country}, {horizon}-year horizon")

    # ── J-Curve simulator ─────────────────────────────────────────────────────
    def _run_jcurve(self):
        if self.ml_model is None:
            messagebox.showwarning("No Model",
                "Train an ML model first in the ML Models tab.")
            return
        if not self._check_data():
            return

        horizon = self.sc_years.get()
        country = self.sc_country.get()
        depth   = self.jc_depth.get() / 100.0
        trough  = max(1, min(self.jc_trough.get(), horizon))
        ltgain  = self.jc_ltgain.get() / 100.0
        target  = (self.ml_target.get()
                   if hasattr(self, "ml_target") and self.ml_target.get()
                   else "Y by L")

        df      = self.df.copy()
        if country != "All Countries":
            df = df[df["Country"] == country]
        last_yr  = df["Year"].max()
        base_df  = df[df["Year"] == last_yr]
        base_val = (base_df[target].mean()
                    if target in base_df.columns else None)

        # J-curve multiplier profile
        # Phase 1 (t ≤ trough): cosine decline 1 → (1 - depth)
        # Phase 2 (t > trough): cosine recovery (1 - depth) → (1 + ltgain)
        multipliers = []
        for t in range(1, horizon + 1):
            if t <= trough:
                frac = t / trough
                m    = 1.0 - depth * (1 - np.cos(np.pi * frac)) / 2
            else:
                frac = (t - trough) / max(1, horizon - trough)
                m    = ((1.0 - depth) +
                        (depth + ltgain) * (1 - np.cos(np.pi * frac)) / 2)
            multipliers.append(m)

        # Baseline ML projection
        proj_rows = []
        for _ in range(1, horizon + 1):
            row_data = {}
            for f in self.ml_features:
                base_f = f.replace("_lag1", "")
                row_data[f] = (base_df[base_f].mean()
                               if base_f in base_df.columns else 0.0)
            proj_rows.append(row_data)
        X_proj      = pd.DataFrame(proj_rows)[self.ml_features]
        X_sc_arr    = self.ml_scaler.transform(X_proj)
        base_preds  = self.ml_model.predict(X_sc_arr)
        jcurve_preds = base_preds * np.array(multipliers)
        years_proj   = [last_yr + i for i in range(1, horizon + 1)]

        clear_txt(self.sc_txt)
        write(self.sc_txt,
              f"J-CURVE POLICY SHOCK SIMULATION\n{'='*52}\n"
              f"Country    : {country}\n"
              f"Base year  : {last_yr}\n"
              f"Horizon    : {horizon} years\n"
              f"Target var : {target}\n"
              f"Shock depth: −{depth*100:.1f}%  at trough year {trough}\n"
              f"LT gain    : +{ltgain*100:.1f}%  by year {horizon}\n\n"
              f"{'Year':>6}  {'Baseline':>12}  {'J-Curve':>12}"
              f"  {'Multiplier':>10}\n{'─'*46}\n")
        for yr, bp, jp, m in zip(years_proj, base_preds,
                                  jcurve_preds, multipliers):
            write(self.sc_txt,
                  f"{yr:>6}  {bp:>12.3f}  {jp:>12.3f}  {m:>10.4f}\n")

        for w in self.sc_plot_frame.winfo_children():
            w.destroy()
        fig  = Figure(figsize=(10, 6), facecolor=theme.BG)
        ax1  = fig.add_subplot(211)
        ax2  = fig.add_subplot(212)

        if base_val is not None:
            ax1.axhline(base_val, color=theme.GRAY, linewidth=1,
                        linestyle=":",
                        label=f"Base ({last_yr}) = {base_val:.3f}")
        ax1.plot(years_proj, base_preds,
                 color=theme.BLUE, linewidth=2,
                 marker="o", markersize=4,
                 label="Baseline ML projection", alpha=0.8)
        ax1.plot(years_proj, jcurve_preds,
                 color=theme.TEAL, linewidth=2.5,
                 marker="s", markersize=5,
                 label="J-Curve path", zorder=4)
        ax1.fill_between(years_proj,
                         np.minimum(base_preds, jcurve_preds),
                         np.maximum(base_preds, jcurve_preds),
                         alpha=0.15, color=theme.AMBER)
        ax1.axvline(last_yr + trough, color=theme.RED, linewidth=1.2,
                    linestyle="--", alpha=0.7,
                    label=f"Trough (year {trough})")
        ax1.set_ylabel(target, color=theme.FG)
        ax1.set_title(f"J-Curve Simulation — {country}", color=theme.FG)
        ax1.legend(facecolor=theme.WBG, labelcolor=theme.FG, fontsize=8)
        ax1.tick_params(colors=theme.FG)

        ax2.plot(years_proj, (np.array(multipliers) - 1) * 100,
                 color=theme.AMBER, linewidth=2, marker="o", markersize=4)
        ax2.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        ax2.axvline(last_yr + trough, color=theme.RED, linewidth=1.2,
                    linestyle="--", alpha=0.7)
        ax2.fill_between(years_proj, 0,
                         (np.array(multipliers) - 1) * 100,
                         alpha=0.2,
                         color=theme.RED if min(multipliers) < 1
                         else theme.TEAL)
        ax2.set_xlabel("Year", color=theme.FG)
        ax2.set_ylabel("Deviation from baseline (%)", color=theme.FG)
        ax2.set_title("Shock multiplier profile", color=theme.FG, fontsize=9)
        ax2.tick_params(colors=theme.FG)

        fig.suptitle(
            f"J-Curve: −{depth*100:.1f}% dip at t+{trough}"
            f" → +{ltgain*100:.1f}% LT gain",
            color=theme.FG, fontsize=11)
        fig.tight_layout()
        embed_figure(fig, self.sc_plot_frame)
        self.status_var.set(
            f"J-Curve complete — trough t+{trough}, "
            f"LT gain +{ltgain*100:.1f}%")
