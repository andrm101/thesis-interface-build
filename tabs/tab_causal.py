"""
tabs/tab_causal.py  —  Tab 14 · Causal: quasi-experimental and causal-ML
designs from causal.py.

  • Distance-to-frontier FE regression (R&D × gap)
  • Panel local projections of an R&D change, linear or state-dependent
  • Staggered event study (Callaway-Sant'Anna), EU accession or custom events
  • Synthetic control with in-space placebos
  • Double machine learning with heterogeneity by frontier gap and group

These designs need LEVEL variables (output per worker, R&D % GDP, frontier
gap) — load data/panel_levels.csv (built by build_panel.py).

Mixin: CausalTabMixin
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

import causal
import gmm
import theme
from constants import INNOVATIVE_CLUSTER
from helpers import make_text, write, clear_txt, embed_figure

LEVEL_PANEL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "panel_levels.csv")
DEFAULTS = {"y": "Y_per_worker", "rd": "RD_pct_GDP", "gap": "Frontier_gap"}
DEFAULT_CONTROLS = ("Savings_rate", "Tertiary_share")
STATES = ("Frontier gap > median (catch-up)", "A-priori Emerging group",
          "None (linear)")
EVENT_SETS = ("EU accession (2004/2007/2013)",
              "R&D tax reforms (OECD subsidy jumps)", "Custom (below)")


class CausalTabMixin:
    # ══════════════════════════════════════════════════════════════════════
    # TAB 14 — CAUSAL
    # ══════════════════════════════════════════════════════════════════════
    def _tab_causal(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=12, pady=6)

        r0 = ttk.Frame(ctrl); r0.pack(fill=tk.X, pady=2)
        self.cz_y   = tk.StringVar(value=DEFAULTS["y"])
        self.cz_rd  = tk.StringVar(value=DEFAULTS["rd"])
        self.cz_gap = tk.StringVar(value=DEFAULTS["gap"])
        self._cz_cbs = []
        for lbl, var in (("Outcome (level):", self.cz_y),
                         ("R&D / treatment:", self.cz_rd),
                         ("Frontier gap:", self.cz_gap)):
            ttk.Label(r0, text=lbl).pack(side=tk.LEFT)
            cb = ttk.Combobox(r0, textvariable=var, state="readonly",
                              width=20)
            cb.pack(side=tk.LEFT, padx=(4, 10))
            self._cz_cbs.append(cb)
        ttk.Button(r0, text="Load level panel",
                   style="Ghost.TButton",
                   command=self._cz_load_levels).pack(side=tk.RIGHT)

        r1 = ttk.Frame(ctrl); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Controls:").pack(side=tk.LEFT)
        self.cz_ctrl_lb = tk.Listbox(
            r1, selectmode=tk.MULTIPLE, height=3, width=34,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            exportselection=False, font=("Segoe UI", 9))
        self.cz_ctrl_lb.pack(side=tk.LEFT, padx=4)
        opts = ttk.Frame(r1); opts.pack(side=tk.LEFT, padx=8)
        self.cz_h      = tk.IntVar(value=6)
        self.cz_state  = tk.StringVar(value=STATES[0])
        self.cz_events = tk.StringVar(value=EVENT_SETS[0])
        self.cz_custom = tk.StringVar(value="Poland:2016, Slovakia:2015")
        self.cz_ctrlgrp = tk.StringVar(value=causal.CONTROL_GROUPS[1])
        self.cz_detrend = tk.BooleanVar(value=False)
        self.cz_learner = tk.StringVar(value="Random Forest")
        self.cz_sc_unit = tk.StringVar(value="Poland")
        self.cz_sc_year = tk.StringVar(value="2004")
        self.cz_sc_demean = tk.BooleanVar(value=True)
        grid = [
            ("LP horizon:", ttk.Spinbox(opts, from_=2, to=10, width=4,
                                        textvariable=self.cz_h)),
            ("LP state:", ttk.Combobox(opts, textvariable=self.cz_state,
                                       values=STATES, state="readonly",
                                       width=30)),
            ("Events:", ttk.Combobox(opts, textvariable=self.cz_events,
                                     values=EVENT_SETS, state="readonly",
                                     width=30)),
            ("Custom events:", ttk.Entry(opts, textvariable=self.cz_custom,
                                         width=32)),
        ]
        for i, (lbl, w) in enumerate(grid):
            ttk.Label(opts, text=lbl).grid(row=i % 2, column=(i // 2) * 2,
                                           sticky=tk.W, padx=(6, 2))
            w.grid(row=i % 2, column=(i // 2) * 2 + 1, sticky=tk.W, pady=1)
        opts2 = ttk.Frame(r1); opts2.pack(side=tk.LEFT, padx=8)
        ttk.Label(opts2, text="ES controls:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(opts2, textvariable=self.cz_ctrlgrp,
                     values=causal.CONTROL_GROUPS, state="readonly",
                     width=15).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(opts2, text="Detrend pre-trends",
                        variable=self.cz_detrend).grid(row=0, column=2,
                                                       padx=4)
        ttk.Label(opts2, text="SC unit / year:").grid(row=1, column=0,
                                                      sticky=tk.W)
        self.cz_sc_cb = ttk.Combobox(opts2, textvariable=self.cz_sc_unit,
                                     state="readonly", width=15)
        self.cz_sc_cb.grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(opts2, textvariable=self.cz_sc_year, width=6).grid(
            row=1, column=2, sticky=tk.W, padx=4)
        ttk.Checkbutton(opts2, text="Demeaned SC",
                        variable=self.cz_sc_demean).grid(row=1, column=3)
        ttk.Label(opts2, text="DML learner:").grid(row=2, column=0,
                                                   sticky=tk.W)
        ttk.Combobox(opts2, textvariable=self.cz_learner,
                     values=("Random Forest", "Lasso"), state="readonly",
                     width=15).grid(row=2, column=1, sticky=tk.W)
        self.gm_method = tk.StringVar(value="System")
        self.gm_ylags  = tk.IntVar(value=2)
        self.gm_lo     = tk.IntVar(value=2)
        self.gm_hi     = tk.IntVar(value=3)
        self.gm_inter  = tk.BooleanVar(value=True)
        ttk.Label(opts2, text="GMM:").grid(row=3, column=0, sticky=tk.W)
        gm = ttk.Frame(opts2); gm.grid(row=3, column=1, columnspan=3,
                                       sticky=tk.W)
        ttk.Combobox(gm, textvariable=self.gm_method, width=10,
                     values=("System", "Difference"),
                     state="readonly").pack(side=tk.LEFT)
        ttk.Label(gm, text=" y lags").pack(side=tk.LEFT)
        ttk.Spinbox(gm, from_=1, to=2, width=3,
                    textvariable=self.gm_ylags).pack(side=tk.LEFT)
        ttk.Label(gm, text=" instr. lags").pack(side=tk.LEFT)
        ttk.Spinbox(gm, from_=2, to=4, width=3,
                    textvariable=self.gm_lo).pack(side=tk.LEFT)
        ttk.Spinbox(gm, from_=2, to=6, width=3,
                    textvariable=self.gm_hi).pack(side=tk.LEFT)
        ttk.Checkbutton(gm, text="× Emerging",
                        variable=self.gm_inter).pack(side=tk.LEFT, padx=4)

        r2 = ttk.Frame(ctrl); r2.pack(fill=tk.X, pady=4)
        for label, cmd, style in (
                ("Frontier Regression", self._cz_frontier, "TButton"),
                ("Local Projections",   self._cz_lp,       "TButton"),
                ("Event Study",         self._cz_event,    "Accent.TButton"),
                ("Synthetic Control",   self._cz_synth,    "TButton"),
                ("Double ML",           self._cz_dml,      "TButton"),
                ("Dynamic GMM",         self._cz_gmm,      "TButton")):
            ttk.Button(r2, text=label, style=style, command=cmd).pack(
                side=tk.LEFT, padx=3)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.cz_txt  = make_text(left, height=30, copy_root=self.root)
        self.cz_plot = right

    def _refresh_causal_vars(self):
        if self.df is None:
            return
        num = [c for c in self.df.select_dtypes("number").columns
               if c != "Year"]
        for cb, (key, var) in zip(self._cz_cbs, (("y", self.cz_y),
                                                  ("rd", self.cz_rd),
                                                  ("gap", self.cz_gap))):
            cb["values"] = num
            if var.get() not in num:
                var.set(DEFAULTS[key] if DEFAULTS[key] in num else
                        (num[0] if num else ""))
        self.cz_ctrl_lb.delete(0, tk.END)
        for c in num:
            self.cz_ctrl_lb.insert(tk.END, c)
            if c in DEFAULT_CONTROLS:
                self.cz_ctrl_lb.select_set(tk.END)
        countries = sorted(self.df["Country"].unique())
        self.cz_sc_cb["values"] = countries
        if self.cz_sc_unit.get() not in countries and countries:
            self.cz_sc_unit.set(countries[0])

    # ── helpers ───────────────────────────────────────────────────────────
    def _cz_load_levels(self):
        if not os.path.exists(LEVEL_PANEL):
            messagebox.showwarning(
                "Level panel", "data/panel_levels.csv not found — run "
                "`python build_panel.py` first.")
            return
        if self.load_path(LEVEL_PANEL):
            self.cz_y.set(DEFAULTS["y"])
            self.cz_rd.set(DEFAULTS["rd"])
            self.cz_gap.set(DEFAULTS["gap"])
            self._refresh_causal_vars()

    def _cz_ready(self, need_gap=False) -> bool:
        if not self._check_data():
            return False
        self._refresh_causal_vars()
        need = [self.cz_y.get(), self.cz_rd.get()] + \
               ([self.cz_gap.get()] if need_gap else [])
        missing = [c for c in need if c not in self.df.columns]
        if len(set(need)) < len(need):
            messagebox.showwarning(
                "Causal", "Outcome, treatment and gap must be different "
                "variables.")
            return False
        if missing or DEFAULTS["y"] not in self.df.columns:
            if missing or not messagebox.askyesno(
                    "Level variables",
                    "The loaded panel has no level variables (growth "
                    "indices only), so causal designs may be meaningless.\n\n"
                    "Continue anyway? (Choose No, then 'Load level panel'.)"):
                return False
        return True

    def _cz_controls(self):
        return [self.cz_ctrl_lb.get(i) for i in self.cz_ctrl_lb.curselection()]

    def _cz_groups(self) -> pd.Series:
        cs = self.df["Country"].unique()
        return pd.Series({c: ("Innovative" if c.lower() in INNOVATIVE_CLUSTER
                              else "Emerging") for c in cs})

    def _cz_events_dict(self) -> dict:
        choice = self.cz_events.get()
        if choice == EVENT_SETS[0]:
            return dict(causal.EU_ACCESSION)
        if choice == EVENT_SETS[1]:
            col = "RD_tax_reform_year"
            if col not in self.df.columns:
                messagebox.showwarning(
                    "Events", "No R&D tax-reform dates in this panel — "
                    "rebuild it with the policy files (build_panel.py).")
                return {}
            ev = (self.df.dropna(subset=[col]).groupby("Country")[col]
                  .first().astype(int))
            return ev.to_dict()
        return causal.parse_events(self.cz_custom.get())

    def _cz_run(self, label, work, show):
        """Run *work* in a thread, then *show(result)* on the Tk thread."""
        self.status_var.set(f"Running {label}…")

        def _t():
            try:
                res = work()
                self.root.after(0, lambda: (show(res),
                                            self.status_var.set(
                                                f"{label} done")))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    f"{label} Error", str(exc)))
        threading.Thread(target=_t, daemon=True).start()

    def _cz_fig(self, w=10, h=6.5):
        for wdg in self.cz_plot.winfo_children():
            wdg.destroy()
        return Figure(figsize=(w, h), facecolor=theme.BG)

    def _cz_show(self, fig):
        fig.tight_layout()
        embed_figure(fig, self.cz_plot)

    @staticmethod
    def _p(p):
        from constants import stars
        return f"{p:.3f}{stars(p)}"

    # ── 1. Frontier ───────────────────────────────────────────────────────
    def _cz_frontier(self):
        if not self._cz_ready(need_gap=True):
            return
        y, rd, gap, ctrls = (self.cz_y.get(), self.cz_rd.get(),
                             self.cz_gap.get(), self._cz_controls())
        df = self.df.copy()

        def show(r):
            L = [f"DISTANCE-TO-FRONTIER GROWTH REGRESSION\n{'='*62}\n",
                 f"  growth = Δ100·log {y};  two-way FE; SE clustered by "
                 f"country\n  N = {r.nobs}, within R² = {r.r2_within:.3f}\n\n",
                 f"  {'Variable':<24}{'Coef':>9}{'SE':>9}{'p':>11}\n"]
            for _, row in r.table.iterrows():
                L.append(f"  {row.Variable:<24}{row.Coef:>9.3f}"
                         f"{row.SE:>9.3f}{self._p(row.p):>11}\n")
            L.append(f"\n  Marginal effect of {rd} on growth by gap:\n")
            for _, m in r.marginal.iterrows():
                L.append(f"   gap q{int(m.Gap_quantile*100):<3} ({m.Gap:.2f})"
                         f"  {m.dGrowth_dRD:>7.3f}  (SE {m.SE:.3f}) "
                         f"p={self._p(m.p)}\n")
            b3 = r.table.set_index("Variable").loc["RDxGap_l1"]
            L.append(
                "\n  RDxGap > 0: R&D pays more far from the frontier "
                "(absorption,\n  Griffith et al. 2004); < 0: more near it "
                "(Acemoglu et al. 2006).\n"
                f"  Here: {b3.Coef:+.3f}, p = {self._p(b3.p)}.\n"
                "  Gap_l1 > 0 is conditional β-convergence.\n")
            clear_txt(self.cz_txt); write(self.cz_txt, "".join(L))
            fig = self._cz_fig()
            ax = fig.add_subplot(111)
            m = r.marginal
            ax.plot(m.Gap, m.dGrowth_dRD, "o-", color=theme.TEAL)
            ax.fill_between(m.Gap, m.dGrowth_dRD - 1.96 * m.SE,
                            m.dGrowth_dRD + 1.96 * m.SE, color=theme.TEAL,
                            alpha=0.2)
            ax.axhline(0, color=theme.FG, lw=0.7)
            ax.set_xlabel(f"{gap} (t−1)  →  further from frontier")
            ax.set_ylabel(f"∂ growth / ∂ {rd}")
            ax.set_title("Marginal effect of R&D by distance to frontier "
                         "(95 % CI)", color=theme.FG)
            self._cz_show(fig)

        self._cz_run("Frontier regression",
                     lambda: causal.frontier_regression(df, y, rd, gap, ctrls),
                     show)

    # ── 2. Local projections ──────────────────────────────────────────────
    def _cz_lp(self):
        if not self._cz_ready(need_gap=self.cz_state.get() == STATES[0]):
            return
        y, rd, gap = self.cz_y.get(), self.cz_rd.get(), self.cz_gap.get()
        H = int(self.cz_h.get())
        df = self.df.copy()
        st = self.cz_state.get()
        if st == STATES[0]:
            lag = df.sort_values(["Country", "Year"]).groupby(
                "Country")[gap].shift(1).reindex(df.index)
            state = (lag > df[gap].median()).astype(float).where(lag.notna())
            names = ("Catch-up (far)", "Near frontier")
        elif st == STATES[1]:
            g = self._cz_groups()
            state = (df["Country"].map(g) == "Emerging").astype(float)
            names = ("Emerging", "Innovative")
        else:
            state, names = None, ("All", "")

        def show(t):
            L = [f"PANEL LOCAL PROJECTIONS — response of 100·log {y}\n"
                 f"{'='*62}\n  shock: Δ{rd} (one unit);  2 lags of Δshock "
                 f"and Δy;  two-way FE;\n  SE clustered by country.  "
                 f"State: {st}\n\n",
                 f"  {'h':>2} {'state':<16}{'β':>8}{'SE':>8}{'p':>11}{'n':>6}\n"]
            for _, r in t.iterrows():
                L.append(f"  {r.h:>2} {r.state:<16}{r.beta:>8.2f}"
                         f"{r.se:>8.2f}{self._p(r.p):>11}{r.n:>6}\n")
            L.append("\n  β_h = % change in the outcome h years after a "
                     "one-unit rise\n  in the shock variable (e.g. +1 pp "
                     "of GDP on R&D).\n")
            clear_txt(self.cz_txt); write(self.cz_txt, "".join(L))
            fig = self._cz_fig()
            ax = fig.add_subplot(111)
            pal = [theme.TEAL, theme.ACCENT, theme.AMBER]
            for i, (nm, s) in enumerate(t.groupby("state", sort=False)):
                off = (i - 0.5) * 0.12 if t.state.nunique() > 1 else 0
                ax.plot(s.h + off, s.beta, "o-", color=pal[i], label=nm)
                ax.fill_between(s.h + off, s.lo, s.hi, color=pal[i],
                                alpha=0.15)
            ax.axhline(0, color=theme.FG, lw=0.7)
            ax.set_xlabel("years after shock (h)")
            ax.set_ylabel(f"% response of {y}")
            ax.set_title("Local-projection impulse responses (95 % CI)",
                         color=theme.FG)
            ax.legend(fontsize=8)
            self._cz_show(fig)

        self._cz_run("Local projections",
                     lambda: causal.local_projections(df, y, rd, H, 2, state,
                                                      names), show)

    # ── 3. Event study ────────────────────────────────────────────────────
    def _cz_event(self):
        if not self._cz_ready():
            return
        y = self.cz_y.get()
        ev = self._cz_events_dict()
        if not ev:
            messagebox.showwarning("Event study", "No valid events "
                                   "(format: Country:Year, …).")
            return
        df, ctrl, dt = self.df.copy(), self.cz_ctrlgrp.get(), \
            self.cz_detrend.get()

        def show(r):
            L = [f"STAGGERED EVENT STUDY (Callaway & Sant'Anna 2021)\n"
                 f"{'='*62}\n  outcome: 100·log {y};  controls: {ctrl}"
                 f"{';  detrended' if dt else ''}\n"
                 f"  treated ({len(r.treated)}): {', '.join(r.treated)}\n"
                 f"  never-treated ({len(r.controls)}): "
                 f"{', '.join(r.controls)}\n\n",
                 f"  {'e':>3}{'ATT':>9}{'SE':>8}{'95% CI':>20}\n"]
            for _, row in r.by_event_time.iterrows():
                L.append(f"  {int(row.e):>3}{row.att:>9.2f}{row.se:>8.2f}"
                         f"   [{row.lo:>6.2f}, {row.hi:>6.2f}]\n")
            L.append(f"\n  Average post-event ATT: {r.overall_post:.2f} % "
                     f"(bootstrap SE {r.overall_post_se:.2f})\n"
                     f"  Pre-trend (mean ATT, e ≤ −2): {r.pretrend_mean:.2f},"
                     f" p = {self._p(r.pretrend_p)}\n")
            if np.isfinite(r.pretrend_p) and r.pretrend_p < 0.05:
                L.append("\n  ⚠ Pre-trends are significant: treated countries"
                         " were already\n  diverging before the event, so "
                         "the ATT is not causal as is.\n  Try 'Not yet "
                         "treated' controls and/or 'Detrend pre-trends'.\n")
            clear_txt(self.cz_txt); write(self.cz_txt, "".join(L))
            fig = self._cz_fig()
            ax = fig.add_subplot(111)
            t = r.by_event_time
            base = pd.DataFrame({"e": [-1], "att": [0.0], "lo": [0.0],
                                 "hi": [0.0]})
            t = pd.concat([t, base]).sort_values("e")
            cols = [theme.GRAY if e < 0 else theme.TEAL for e in t.e]
            ax.vlines(t.e, t.lo, t.hi, colors=cols, lw=2)
            ax.scatter(t.e, t.att, c=cols, zorder=3)
            ax.axhline(0, color=theme.FG, lw=0.7)
            ax.axvline(-0.5, color=theme.RED, lw=0.8, ls="--")
            ax.set_xlabel("years relative to event (base = −1)")
            ax.set_ylabel(f"ATT, % of {y}")
            ax.set_title("Event-study ATT by event time (95 % bootstrap CI)",
                         color=theme.FG)
            self._cz_show(fig)

        self._cz_run("Event study",
                     lambda: causal.event_study(df, y, ev, 4, 8, 299,
                                                control=ctrl, detrend=dt),
                     show)

    # ── 4. Synthetic control ──────────────────────────────────────────────
    def _cz_synth(self):
        if not self._cz_ready():
            return
        y, unit = self.cz_y.get(), self.cz_sc_unit.get()
        try:
            year = int(self.cz_sc_year.get())
        except ValueError:
            messagebox.showwarning("Synthetic control", "Year must be an "
                                   "integer.")
            return
        ev = self._cz_events_dict()
        # Donors: countries never hit by an event in the chosen set.
        donors = [c for c in self.df["Country"].unique()
                  if c not in ev and c != unit]
        df, dm = self.df.copy(), self.cz_sc_demean.get()

        def show(r):
            L = [f"SYNTHETIC CONTROL — {r.treated}, treatment {r.year}\n"
                 f"{'='*62}\n  outcome: 100·log {y}"
                 f"{' (pre-period demeaned)' if dm else ''}\n"
                 f"  donors ({len(r.weights)}): countries without an event "
                 f"in the chosen set\n\n  Weights:\n"]
            for c, w in r.weights[r.weights > 0.005].items():
                L.append(f"    {c:<16}{w:>7.3f}\n")
            L.append(f"\n  Pre RMSPE {r.pre_rmspe:.2f}   Post RMSPE "
                     f"{r.post_rmspe:.2f}   ratio {r.ratio:.2f}\n"
                     f"  Average post gap: {r.avg_post_gap:+.2f} %\n"
                     f"  Placebo p-value (rank of post/pre RMSPE ratio "
                     f"among {len(r.placebo_ratios)}\n  in-space placebos): "
                     f"{r.p_value:.3f}\n")
            clear_txt(self.cz_txt); write(self.cz_txt, "".join(L))
            fig = self._cz_fig(10, 7)
            a1 = fig.add_subplot(211)
            p = r.path
            a1.plot(p.Year, p.actual, color=theme.TEAL, lw=2, label=unit)
            a1.plot(p.Year, p.synthetic, color=theme.AMBER, lw=2, ls="--",
                    label="Synthetic")
            a1.axvline(r.year - 0.5, color=theme.RED, lw=0.8, ls="--")
            a1.legend(fontsize=8)
            a1.set_title(f"{unit} vs synthetic control", color=theme.FG)
            a2 = fig.add_subplot(212)
            a2.plot(p.Year, p.gap, color=theme.TEAL, lw=2)
            a2.axhline(0, color=theme.FG, lw=0.7)
            a2.axvline(r.year - 0.5, color=theme.RED, lw=0.8, ls="--")
            a2.set_title("Gap (treated − synthetic)", color=theme.FG)
            self._cz_show(fig)

        self._cz_run("Synthetic control",
                     lambda: causal.synthetic_control(df, y, unit, year,
                                                      donors, dm), show)

    # ── 5. Double ML ──────────────────────────────────────────────────────
    def _cz_dml(self):
        if not self._cz_ready(need_gap=True):
            return
        y, rd, gap = self.cz_y.get(), self.cz_rd.get(), self.cz_gap.get()
        ctrls = [c for c in self._cz_controls() if c not in (y, rd)]
        if gap not in ctrls:
            ctrls.append(gap)
        df, learner, groups = self.df.copy(), self.cz_learner.get(), \
            self._cz_groups()

        def show(r):
            L = [f"DOUBLE MACHINE LEARNING — partially linear model\n"
                 f"{'='*62}\n  growth = Δ100·log {y};  treatment = {rd}"
                 f"(t−1)\n  nuisances: {learner}, cross-fitted by country "
                 f"(GroupKFold)\n  controls: lagged {', '.join(ctrls)}, "
                 f"their country means, lagged growth, year\n"
                 f"  nuisance R²: outcome {r.nuisance_r2['outcome']:.3f}, "
                 f"treatment {r.nuisance_r2['treatment']:.3f}\n\n"
                 f"  θ (avg effect of +1 unit {rd} on growth) = "
                 f"{r.theta:.3f}  (SE {r.se:.3f}), p = {self._p(r.p)}\n"
                 f"  N = {r.n}\n"]
            if r.cate_slope is not None:
                pz = 2 * (1 - stats.norm.cdf(abs(r.cate_slope / r.cate_se)))
                L.append(f"\n  Heterogeneity: ∂θ/∂{gap} = {r.cate_slope:.3f}"
                         f" (SE {r.cate_se:.3f}), p = {self._p(pz)}\n"
                         "  < 0 ⇒ R&D pays more near the frontier.\n")
            if len(r.by_group):
                L.append("\n  By group (a-priori thesis clusters):\n")
                for _, g in r.by_group.iterrows():
                    L.append(f"    {g.Group:<12} θ = {g.theta:>7.3f} "
                             f"(SE {g.se:.3f}) p = {self._p(g.p)}  "
                             f"[{g.countries} countries, n={g.n}]\n")
            clear_txt(self.cz_txt); write(self.cz_txt, "".join(L))
            fig = self._cz_fig(9, 5.5)
            ax = fig.add_subplot(111)
            rows = [("All", r.theta, r.se)] + [
                (g.Group, g.theta, g.se) for _, g in r.by_group.iterrows()]
            xs = np.arange(len(rows))
            ax.bar(xs, [v for _, v, _ in rows],
                   yerr=[1.96 * s for *_, s in rows], capsize=6,
                   color=[theme.BLUE, theme.TEAL, theme.ACCENT][:len(rows)])
            ax.set_xticks(xs, [n for n, *_ in rows])
            ax.axhline(0, color=theme.FG, lw=0.7)
            ax.set_ylabel(f"θ: growth effect of +1 {rd}")
            ax.set_title("DML treatment effect (95 % CI)", color=theme.FG)
            self._cz_show(fig)

        self._cz_run("Double ML",
                     lambda: causal.dml_plr(df, y, rd, ctrls, gap, groups,
                                            learner), show)

    # ── 6. Dynamic panel GMM ──────────────────────────────────────────────
    def _cz_gmm(self):
        if not self._cz_ready():
            return
        y, rd = self.cz_y.get(), self.cz_rd.get()
        ctrls = [c for c in self._cz_controls() if c not in (y, rd)]
        lo, hi = int(self.gm_lo.get()), int(self.gm_hi.get())
        if hi < lo:
            messagebox.showwarning("GMM", "Instrument lag range: upper < lower.")
            return
        method = self.gm_method.get().lower()
        ylags, inter = int(self.gm_ylags.get()), self.gm_inter.get()
        df = self.df.copy()
        ycol = f"100log_{y}"
        df[ycol] = 100 * np.log(df[y].where(df[y] > 0))
        endog = [rd]
        if inter:
            g = self._cz_groups()
            df[f"{rd} × Emerging"] = df[rd] * (
                df["Country"].map(g) == "Emerging").astype(float)
            endog.append(f"{rd} × Emerging")

        def show(r):
            L = [f"DYNAMIC PANEL GMM — {r.method}\n{'='*62}\n"
                 f"  y = 100·log {y};  endogenous: {', '.join(endog)}\n"
                 f"  predetermined: {', '.join(ctrls) or '—'}\n"
                 f"  {r.notes[0]}\n  groups N = {r.n_groups}, obs = "
                 f"{r.n_obs}, instruments = {r.n_instruments}\n\n",
                 f"  {'Variable':<34}{'Coef':>9}{'SE':>9}{'p':>11}\n"]
            for _, row in r.table().iterrows():
                L.append(f"  {row.Variable[:33]:<34}{row.Coef:>9.3f}"
                         f"{row.SE:>9.3f}{self._p(row.p):>11}\n")
            L.append("\n  Long-run effects β/(1−Σρ):\n")
            for k_, (e, se) in r.long_run.items():
                p = 2 * (1 - stats.norm.cdf(abs(e / se))) if se else np.nan
                L.append(f"    {k_[:32]:<33}{e:>9.2f}  (SE {se:.2f}) "
                         f"p={self._p(p)}\n")
            ok = lambda c: "✓" if c else "✗"  # noqa: E731
            L.append(
                "\n  Diagnostics:\n"
                f"    AR(1) p = {r.ar1_p:.3f}  {ok(r.ar1_p < 0.10)} should "
                "reject (differenced errors are MA(1))\n"
                f"    AR(2) p = {r.ar2_p:.3f}  {ok(r.ar2_p > 0.05)} should "
                "NOT reject — else raise y lags or instrument lags\n"
                f"    Hansen J({r.hansen_df}) p = {r.hansen_p:.3f}  "
                f"{ok(0.05 < r.hansen_p < 0.99)} should not reject; ≈ 1 "
                "signals too many instruments\n"
                f"    Instruments {r.n_instruments} vs groups {r.n_groups}  "
                f"{ok(r.n_instruments <= r.n_groups)}\n")
            for n in r.notes[1:]:
                L.append(f"  {n}\n")
            clear_txt(self.cz_txt); write(self.cz_txt, "".join(L))
            fig = self._cz_fig(9, 5)
            ax = fig.add_subplot(111)
            t = r.table()
            t = t[~t.Variable.isin(["const"])]
            ax.errorbar(t.Coef, np.arange(len(t)), xerr=1.96 * t.SE,
                        fmt="o", color=theme.TEAL, ecolor=theme.FG, capsize=4)
            ax.set_yticks(np.arange(len(t)), t.Variable)
            ax.axvline(0, color=theme.RED, lw=0.8, ls="--")
            ax.set_title(f"{r.method}: coefficients (95 % CI)",
                         color=theme.FG)
            ax.tick_params(labelsize=8)
            self._cz_show(fig)

        self._cz_run("Dynamic GMM",
                     lambda: gmm.estimate(df, ycol, endog=endog,
                                          predet=ctrls, method=method,
                                          lags=(lo, hi), y_lags=ylags),
                     show)

