"""
tabs/tab_stats.py  —  Descriptive statistics, correlation heatmap, distributions.
Mixin: StatsTabMixin
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

import theme
from helpers import make_text, write, clear_txt, embed_figure


class StatsTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_stats(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Summary Statistics",
                   command=self._run_summary).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Correlation Heatmap",
                   command=self._run_corr).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Distributions",
                   command=self._run_dist).pack(side=tk.LEFT, padx=4)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.stats_txt        = make_text(left, height=30, copy_root=self.root)
        self.stats_plot_frame = right

    def _run_summary(self):
        if not self._check_data():
            return
        df   = self.df.select_dtypes(include="number")
        desc = df.describe().T
        desc["cv"] = (desc["std"] / desc["mean"].abs()).round(4)
        lines = [
            f"DESCRIPTIVE STATISTICS\n{'='*68}\n",
            f"{'Variable':<28} {'Mean':>9} {'Std':>9} {'Min':>9} "
            f"{'Median':>9} {'Max':>9} {'CV':>7}\n",
            f"{'─'*68}\n",
        ]
        for var, row in desc.iterrows():
            lines.append(
                f"{var:<28} {row['mean']:>9.3f} {row['std']:>9.3f} "
                f"{row['min']:>9.3f} {row['50%']:>9.3f} "
                f"{row['max']:>9.3f} {row['cv']:>7.3f}\n")
        clear_txt(self.stats_txt)
        write(self.stats_txt, "".join(lines))

    def _run_corr(self):
        if not self._check_data():
            return
        for w in self.stats_plot_frame.winfo_children():
            w.destroy()
        plot_cols = [c for c in self.df.select_dtypes("number").columns
                     if c != "Year"]
        corr = self.df[plot_cols].corr()
        fig  = Figure(figsize=(10, 8), facecolor=theme.BG)
        ax   = fig.add_subplot(111)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                    fmt=".2f", linewidths=0.4, ax=ax,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix (lower triangle)",
                     color=theme.FG, pad=12)
        ax.tick_params(colors=theme.FG, labelsize=8)
        fig.tight_layout()
        embed_figure(fig, self.stats_plot_frame)

    def _run_dist(self):
        if not self._check_data():
            return
        for w in self.stats_plot_frame.winfo_children():
            w.destroy()
        pref = ["Y by L", "PIB towards research", "Labor in research",
                "Human Capital Proxy", "Savings Percentage", "Patents per capita"]
        cols = [c for c in pref if c in self.df.columns]
        if len(cols) < 6:
            cols += [c for c in self.df.select_dtypes("number").columns
                     if c not in cols and c != "Year"]
        cols = cols[:6]
        fig = Figure(figsize=(12, 8), facecolor=theme.BG)
        for i, col in enumerate(cols, 1):
            ax   = fig.add_subplot(2, 3, i)
            data = self.df[col].dropna()
            ax.hist(data, bins=20, color=theme.BLUE,
                    edgecolor=theme.DARK, alpha=0.85)
            ax.set_title(col, color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=7)
        fig.suptitle("Variable Distributions", color=theme.FG, fontsize=12)
        fig.tight_layout()
        embed_figure(fig, self.stats_plot_frame)
