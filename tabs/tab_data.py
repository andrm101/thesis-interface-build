"""
tabs/tab_data.py — Data loading, filtering, preview, and transformation.

Thesis-aligned features
-----------------------
  • Loads balanced or unbalanced EU panels (country × year long-form).
  • One-click filters for the thesis sample: EU-25, Innovative cluster,
    Emerging cluster, plus legacy West/East splits.
  • Transformations applied in-place as new columns: growth rate, first
    difference, log-difference, log-level.
  • Hodrick-Prescott filter plot (trend + cycle) and rolling statistics
    plot (rolling mean ± std band).
  • Panel-structure diagnostic: balanced-ness, observations per country
    per year, and missingness report per variable.

Mixin: DataTabMixin
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.tsa.filters.hp_filter import hpfilter

import theme
from constants import (
    WESTERN_EU, EASTERN_EU,
    INNOVATIVE_CLUSTER, EMERGING_CLUSTER, EU25,
)
from helpers import make_text, write, clear_txt, embed_figure


class DataTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_data(self, parent):
        # ── Top controls: file + filters ──────────────────────────────────
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=12, pady=8)

        # File row
        file_frm = ttk.LabelFrame(top, text="Dataset", padding=8)
        file_frm.pack(fill=tk.X, pady=4)
        self.file_var = tk.StringVar(value="No file selected")
        _fl = ttk.Label(file_frm, textvariable=self.file_var,
                        foreground=theme.GRAY)
        _fl.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        self._special_labels.append((_fl, "GRAY"))
        ttk.Button(file_frm, text="Browse…",
                   command=self._load_file).pack(side=tk.RIGHT, padx=4)

        # Filter row
        flt = ttk.Frame(top)
        flt.pack(fill=tk.X, pady=4)

        yr = ttk.LabelFrame(flt, text="Year Range", padding=8)
        yr.pack(side=tk.LEFT, padx=4, fill=tk.Y)
        ttk.Label(yr, text="From:").grid(row=0, column=0, sticky=tk.W)
        self.year_from = ttk.Entry(yr, width=6)
        self.year_from.grid(row=0, column=1, padx=4)
        ttk.Label(yr, text="To:").grid(row=0, column=2, sticky=tk.W)
        self.year_to = ttk.Entry(yr, width=6)
        self.year_to.grid(row=0, column=3, padx=4)

        cs = ttk.LabelFrame(flt, text="Countries", padding=8)
        cs.pack(side=tk.LEFT, padx=4, fill=tk.BOTH, expand=True)
        sb_c = ttk.Scrollbar(cs)
        sb_c.pack(side=tk.RIGHT, fill=tk.Y)
        self.country_lb = tk.Listbox(
            cs, selectmode=tk.MULTIPLE,
            bg=theme.WBG, fg=theme.FG,
            selectbackground=theme.BLUE, activestyle="dotbox",
            font=("Segoe UI", 9), height=6,
            yscrollcommand=sb_c.set, exportselection=False,
        )
        self.country_lb.pack(fill=tk.BOTH, expand=True)
        sb_c.config(command=self.country_lb.yview)
        btn_row = ttk.Frame(cs)
        btn_row.pack(fill=tk.X, pady=2)
        preset_buttons = [
            ("All",        self._sel_all,    "TButton"),
            ("None",       self._sel_none,   "TButton"),
            ("EU-25",      self._sel_eu25,   "Ghost.TButton"),
            ("Innovative", self._sel_inno,   "Ghost.TButton"),
            ("Emerging",   self._sel_emerg,  "Ghost.TButton"),
            ("West EU",    self._sel_west,   "TButton"),
            ("East EU",    self._sel_east,   "TButton"),
        ]
        for label, cmd, style in preset_buttons:
            ttk.Button(btn_row, text=label, command=cmd,
                       style=style, width=10).pack(side=tk.LEFT, padx=2)

        actions = ttk.Frame(top)
        actions.pack(pady=6)
        ttk.Button(actions, text="Apply Filters", style="Accent.TButton",
                   command=self._apply_filters).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Panel Structure Report",
                   style="Ghost.TButton",
                   command=self._panel_structure_report).pack(
            side=tk.LEFT, padx=4)

        # ── Vertical paned window: preview (top) / tools (bottom) ─────────
        vpane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        vpane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Preview pane
        prev_outer = ttk.Frame(vpane)
        vpane.add(prev_outer, weight=3)

        prev = ttk.LabelFrame(prev_outer,
                              text="Data Preview  (first 100 rows)",
                              padding=6)
        prev.pack(fill=tk.BOTH, expand=True)
        self.preview_tree = ttk.Treeview(prev, show="headings", height=10)
        sb_h = ttk.Scrollbar(prev, orient=tk.HORIZONTAL,
                             command=self.preview_tree.xview)
        sb_v = ttk.Scrollbar(prev, orient=tk.VERTICAL,
                             command=self.preview_tree.yview)
        self.preview_tree.configure(xscrollcommand=sb_h.set,
                                    yscrollcommand=sb_v.set)
        sb_h.pack(side=tk.BOTTOM, fill=tk.X)
        sb_v.pack(side=tk.RIGHT,  fill=tk.Y)
        self.preview_tree.pack(fill=tk.BOTH, expand=True)

        self.data_info = tk.StringVar(value="")
        _di = ttk.Label(prev_outer, textvariable=self.data_info,
                        foreground=theme.TEAL)
        _di.pack(padx=4, pady=2, anchor=tk.W)
        self._special_labels.append((_di, "TEAL"))

        # ── Data Transformation Tools pane ────────────────────────────────
        tools_outer = ttk.Frame(vpane)
        vpane.add(tools_outer, weight=2)
        self._build_data_tools(tools_outer)

    # ── Data Tools UI builder ─────────────────────────────────────────────
    def _build_data_tools(self, parent):
        tools = ttk.LabelFrame(parent,
                               text="Data Transformation Tools",
                               padding=8)
        tools.pack(fill=tk.X, pady=4)

        # Row 1: variable + country + transform
        r1 = ttk.Frame(tools); r1.pack(fill=tk.X, pady=3)
        ttk.Label(r1, text="Variable:").pack(side=tk.LEFT)
        self.tool_var = ttk.Combobox(r1, width=22, state="readonly")
        self.tool_var.pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="Country:").pack(side=tk.LEFT, padx=(12, 0))
        self.tool_country = ttk.Combobox(r1, width=18, state="readonly")
        self.tool_country.pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="Transform:").pack(side=tk.LEFT, padx=(12, 0))
        self.tool_transform = ttk.Combobox(r1, width=20, state="readonly",
                                           values=[
            "Growth Rate (%)",
            "First Difference",
            "Log-Difference",
            "Log-Level",
        ])
        self.tool_transform.current(0)
        self.tool_transform.pack(side=tk.LEFT, padx=4)

        # Row 2: HP lambda + rolling window + buttons
        r2 = ttk.Frame(tools); r2.pack(fill=tk.X, pady=3)
        ttk.Label(r2, text="HP λ:").pack(side=tk.LEFT)
        self.tool_hp_lambda = tk.IntVar(value=100)
        _sp_hp = tk.Spinbox(r2, from_=6, to=100000, increment=100,
                            width=8, textvariable=self.tool_hp_lambda,
                            bg=theme.WBG, fg=theme.FG,
                            insertbackground=theme.FG)
        _sp_hp.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_sp_hp)
        _hp_hint = ttk.Label(r2, text="(100=annual, 1600=quarterly)",
                             foreground=theme.GRAY)
        _hp_hint.pack(side=tk.LEFT, padx=2)
        self._special_labels.append((_hp_hint, "GRAY"))

        ttk.Label(r2, text="   Rolling window:").pack(side=tk.LEFT)
        self.tool_rolling = tk.IntVar(value=3)
        _sp_roll = tk.Spinbox(r2, from_=2, to=20, width=4,
                              textvariable=self.tool_rolling,
                              bg=theme.WBG, fg=theme.FG,
                              insertbackground=theme.FG)
        _sp_roll.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_sp_roll)

        # Row 3: action buttons
        r3 = ttk.Frame(tools); r3.pack(fill=tk.X, pady=3)
        ttk.Button(r3, text="Apply Transform → DataFrame",
                   style="Accent.TButton",
                   command=self._apply_transform).pack(side=tk.LEFT, padx=4)
        ttk.Button(r3, text="HP Filter Plot",
                   command=self._plot_hp_filter).pack(side=tk.LEFT, padx=4)
        ttk.Button(r3, text="Rolling Statistics Plot",
                   command=self._plot_rolling).pack(side=tk.LEFT, padx=4)
        ttk.Button(r3, text="Time Series Plot",
                   command=self._plot_series).pack(side=tk.LEFT, padx=4)

        # Plot frame for data tools output
        self.tools_plot_frame = ttk.Frame(parent)
        self.tools_plot_frame.pack(fill=tk.BOTH, expand=True, pady=2)

    # ── File loading ──────────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select Panel Dataset",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            df = (pd.read_excel(path)
                  if path.lower().endswith((".xlsx", ".xls"))
                  else pd.read_csv(path))
            self.df_raw = df
            self.file_var.set(os.path.basename(path))
            if "Year" in df.columns:
                self.year_from.delete(0, tk.END)
                self.year_from.insert(0, str(int(df["Year"].min())))
                self.year_to.delete(0, tk.END)
                self.year_to.insert(0, str(int(df["Year"].max())))
            if "Country" in df.columns:
                countries = sorted(df["Country"].dropna().unique())
                self.country_lb.delete(0, tk.END)
                for c in countries:
                    self.country_lb.insert(tk.END, c)
                self.country_lb.select_set(0, tk.END)
            self._apply_filters()
            self.status_var.set(
                f"Loaded: {len(df):,} rows · {len(df.columns)} columns")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _apply_filters(self):
        if self.df_raw is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        df = self.df_raw.copy()
        try:
            y0 = int(self.year_from.get())
            y1 = int(self.year_to.get())
            df = df[df["Year"].between(y0, y1)]
        except ValueError:
            pass
        sel = [self.country_lb.get(i) for i in self.country_lb.curselection()]
        if sel:
            df = df[df["Country"].isin(sel)]
        self.df = df.reset_index(drop=True)
        self._update_preview()
        nc = df["Country"].nunique() if "Country" in df.columns else "?"
        ny = df["Year"].nunique()    if "Year"    in df.columns else "?"
        cols_preview = df.select_dtypes("number").columns.tolist()[:6]
        self.data_info.set(
            f"{len(df):,} observations  ·  {nc} countries  ·  {ny} years  ·  "
            f"Numeric cols: {cols_preview} …")
        self.status_var.set(f"Ready — {len(df):,} obs · {nc} countries")
        self._refresh_tool_vars()

    def _update_preview(self):
        t = self.preview_tree
        t.delete(*t.get_children())
        if self.df is None or self.df.empty:
            return
        cols = list(self.df.columns)
        t["columns"] = cols
        for c in cols:
            t.heading(c, text=c)
            t.column(c, width=max(80, len(c) * 8), anchor=tk.CENTER)
        for _, row in self.df.head(100).iterrows():
            t.insert("", tk.END, values=[
                f"{v:.3f}" if isinstance(v, float) else v for v in row])

    # ── Country selection helpers ─────────────────────────────────────────
    def _sel_all(self):  self.country_lb.select_set(0, tk.END)
    def _sel_none(self): self.country_lb.selection_clear(0, tk.END)

    def _sel_group(self, group):
        self.country_lb.selection_clear(0, tk.END)
        for i in range(self.country_lb.size()):
            if self.country_lb.get(i).lower() in group:
                self.country_lb.select_set(i)

    def _sel_west(self):  self._sel_group(WESTERN_EU)
    def _sel_east(self):  self._sel_group(EASTERN_EU)
    def _sel_eu25(self):  self._sel_group(EU25)
    def _sel_inno(self):  self._sel_group(INNOVATIVE_CLUSTER)
    def _sel_emerg(self): self._sel_group(EMERGING_CLUSTER)

    # ── Panel structure diagnostic ────────────────────────────────────────
    def _panel_structure_report(self):
        """Pop a dialog summarising balanced-ness and missingness."""
        if not self._check_data():
            return
        df = self.df
        if "Country" not in df.columns or "Year" not in df.columns:
            messagebox.showwarning(
                "Panel structure",
                "Need 'Country' and 'Year' columns to analyse panel shape.")
            return
        n_ctry  = df["Country"].nunique()
        n_yrs   = df["Year"].nunique()
        n_obs   = len(df)
        years   = sorted(df["Year"].unique())
        expected = n_ctry * n_yrs
        balanced = (n_obs == expected)
        per_ctry = df.groupby("Country")["Year"].nunique()
        min_yrs, max_yrs = int(per_ctry.min()), int(per_ctry.max())
        num_cols = [c for c in df.select_dtypes("number").columns
                    if c != "Year"]
        missing = (df[num_cols].isna().mean() * 100).round(2).sort_values(
            ascending=False)
        lines = [
            f"PANEL STRUCTURE — {n_ctry} countries × {n_yrs} years",
            f"Time span        : {min(years)}–{max(years)}",
            f"Total rows       : {n_obs:,}   (expected if balanced: "
            f"{expected:,})",
            f"Balanced panel?  : {'YES' if balanced else 'NO'}",
            f"Years per country: min={min_yrs}, max={max_yrs}",
            "",
            "Missingness (% of rows with NaN) — top 10 numeric variables:",
        ]
        for col, pct in missing.head(10).items():
            lines.append(f"   {col:<32} {pct:>6.2f} %")
        messagebox.showinfo("Panel structure report", "\n".join(lines))

    # ── Data Tools: variable refresh ─────────────────────────────────────
    def _refresh_tool_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.tool_var["values"] = num_cols
        if not self.tool_var.get() or self.tool_var.get() not in num_cols:
            self.tool_var.set(
                "Y by L" if "Y by L" in num_cols
                else (num_cols[0] if num_cols else ""))
        countries = ["All Countries"] + sorted(
            self.df["Country"].unique().tolist()
            if "Country" in self.df.columns else [])
        self.tool_country["values"] = countries
        if not self.tool_country.get():
            self.tool_country.set("All Countries")

    def _get_tool_series(self) -> pd.DataFrame | None:
        """Return the time series for the selected variable and country."""
        if not self._check_data():
            return None
        var     = self.tool_var.get()
        country = self.tool_country.get()
        if not var:
            messagebox.showwarning("Config", "Select a variable.")
            return None
        df = self.df[["Country", "Year", var]].dropna()
        if country != "All Countries":
            df = df[df["Country"] == country]
        return df.sort_values("Year")

    # ── Apply Transform ───────────────────────────────────────────────────
    def _apply_transform(self):
        if not self._check_data():
            return
        var  = self.tool_var.get()
        kind = self.tool_transform.get()
        if not var:
            messagebox.showwarning("Config", "Select a variable.")
            return
        df     = self.df.sort_values(["Country", "Year"]).copy()
        grp    = df.groupby("Country")[var]
        suffix_map = {
            "Growth Rate (%)":  "_grwth",
            "First Difference": "_fd",
            "Log-Difference":   "_logfd",
            "Log-Level":        "_log",
        }
        new_col = var + suffix_map.get(kind, "_tx")
        if kind == "Growth Rate (%)":
            df[new_col] = grp.pct_change() * 100
        elif kind == "First Difference":
            df[new_col] = grp.diff()
        elif kind == "Log-Difference":
            df[new_col] = np.log(df[var].clip(lower=1e-9)).groupby(
                df["Country"]).diff()
        elif kind == "Log-Level":
            df[new_col] = np.log(df[var].clip(lower=1e-9))
        else:
            messagebox.showwarning("Transform", f"Unknown transform: {kind}")
            return
        self.df = df.reset_index(drop=True)
        self._update_preview()
        self._refresh_tool_vars()
        n_new = df[new_col].notna().sum()
        self.status_var.set(
            f"Added column '{new_col}' — {n_new:,} non-null values.")
        messagebox.showinfo("Transform Applied",
                            f"Created column: {new_col}\n"
                            f"Non-null values: {n_new:,}")

    # ── HP Filter plot ────────────────────────────────────────────────────
    def _plot_hp_filter(self):
        df = self._get_tool_series()
        if df is None:
            return
        var     = self.tool_var.get()
        country = self.tool_country.get()
        lam     = self.tool_hp_lambda.get()
        for w in self.tools_plot_frame.winfo_children():
            w.destroy()

        if country == "All Countries":
            # Average across countries per year
            series = df.groupby("Year")[var].mean().dropna()
        else:
            series = df.set_index("Year")[var].dropna()

        if len(series) < 4:
            messagebox.showwarning("HP Filter",
                "Need at least 4 observations.")
            return

        cycle, trend = hpfilter(series.values, lamb=lam)
        years        = series.index.values

        fig  = Figure(figsize=(11, 6.2), facecolor=theme.BG)
        gs   = fig.add_gridspec(3, 1, hspace=0.35)
        ax1  = fig.add_subplot(gs[:2, 0])
        ax2  = fig.add_subplot(gs[2, 0], sharex=ax1)

        # Panel 1: original vs trend
        ax1.plot(years, series.values, color=theme.BLUE,
                 linewidth=1.6, label="Observed", alpha=0.85)
        ax1.plot(years, trend, color=theme.TEAL,
                 linewidth=2.4, label=f"HP Trend  (λ={lam:,})")
        ax1.fill_between(years, series.values, trend,
                         where=(series.values >= trend),
                         color=theme.TEAL, alpha=0.08, interpolate=True)
        ax1.fill_between(years, series.values, trend,
                         where=(series.values <  trend),
                         color=theme.RED,  alpha=0.08, interpolate=True)
        ax1.set_ylabel(var)
        ax1.set_title(
            f"Hodrick–Prescott Decomposition · {country} · {var}",
            pad=10)
        ax1.legend(loc="best")
        ax1.spines["bottom"].set_visible(False)
        ax1.tick_params(axis="x", labelbottom=False)

        # Panel 2: cycle
        ax2.bar(years, cycle,
                color=np.where(cycle >= 0, theme.TEAL, theme.RED),
                alpha=0.8, width=0.85, edgecolor="none")
        ax2.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Cycle")
        ax2.set_title("Cycle component (deviation from trend)",
                      fontsize=10, loc="left")

        fig.tight_layout()
        embed_figure(fig, self.tools_plot_frame)
        self.status_var.set(
            f"HP Filter plotted — {country}, λ={lam:,}")

    # ── Rolling statistics plot ───────────────────────────────────────────
    def _plot_rolling(self):
        df = self._get_tool_series()
        if df is None:
            return
        var     = self.tool_var.get()
        country = self.tool_country.get()
        window  = self.tool_rolling.get()
        for w in self.tools_plot_frame.winfo_children():
            w.destroy()

        if country == "All Countries":
            series = df.groupby("Year")[var].mean().dropna()
        else:
            series = df.set_index("Year")[var].dropna()

        if len(series) < window:
            messagebox.showwarning("Rolling",
                f"Need at least {window} observations.")
            return

        roll_mean = series.rolling(window, center=True).mean()
        roll_std  = series.rolling(window, center=True).std()
        years     = series.index.values

        fig = Figure(figsize=(10, 5), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        ax.plot(years, series.values, color=theme.BLUE,
                linewidth=1.3, alpha=0.55, label="Observed",
                marker="o", markersize=3)
        ax.plot(years, roll_mean, color=theme.TEAL,
                linewidth=2.5, label=f"Rolling Mean (w={window})")
        ax.fill_between(years,
                        roll_mean - roll_std,
                        roll_mean + roll_std,
                        alpha=0.18, color=theme.TEAL,
                        label="±1 Rolling Std")
        ax.set_xlabel("Year")
        ax.set_ylabel(var)
        ax.set_title(
            f"Rolling Statistics · {country} · window = {window} yrs",
            pad=10)
        ax.legend(loc="best")
        fig.tight_layout()
        embed_figure(fig, self.tools_plot_frame)
        self.status_var.set(
            f"Rolling stats plotted — window={window}")

    # ── Time series (multi-country) plot ─────────────────────────────────
    def _plot_series(self):
        if not self._check_data():
            return
        var     = self.tool_var.get()
        country = self.tool_country.get()
        if not var:
            messagebox.showwarning("Config", "Select a variable.")
            return
        for w in self.tools_plot_frame.winfo_children():
            w.destroy()

        df = self.df[["Country", "Year", var]].dropna()
        if country != "All Countries":
            df = df[df["Country"] == country]

        countries = sorted(df["Country"].unique())
        # Highlight Innovative vs Emerging using thesis cluster definitions
        import matplotlib as mpl
        from constants import INNOVATIVE_CLUSTER, EMERGING_CLUSTER
        _get_cmap = getattr(mpl, "colormaps", None)
        if _get_cmap is not None:
            inno_cmap  = mpl.colormaps.get_cmap("YlGnBu")
            emerg_cmap = mpl.colormaps.get_cmap("YlOrRd")
        else:  # older matplotlib
            from matplotlib import cm
            inno_cmap  = cm.get_cmap("YlGnBu")
            emerg_cmap = cm.get_cmap("YlOrRd")
        fallback_pal = [theme.BLUE, theme.TEAL, theme.AMBER, theme.RED,
                        theme.GRAY, "#9b59b6", "#e67e22", "#1abc9c",
                        "#e91e63", "#00bcd4", "#8bc34a", "#ff5722"]

        inno_ctries  = [c for c in countries
                        if c.lower() in INNOVATIVE_CLUSTER]
        emerg_ctries = [c for c in countries
                        if c.lower() in EMERGING_CLUSTER]
        other        = [c for c in countries
                        if c not in inno_ctries + emerg_ctries]

        def pick_color(c, idx):
            if c in inno_ctries:
                i = inno_ctries.index(c)
                return inno_cmap(0.35 + 0.55 * i / max(len(inno_ctries) - 1, 1))
            if c in emerg_ctries:
                i = emerg_ctries.index(c)
                return emerg_cmap(0.35 + 0.55 * i / max(len(emerg_ctries) - 1, 1))
            return fallback_pal[idx % len(fallback_pal)]

        fig = Figure(figsize=(11, 5.2), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        for i, c in enumerate(countries):
            sub = df[df["Country"] == c].sort_values("Year")
            ax.plot(sub["Year"], sub[var],
                    label=c, color=pick_color(c, i),
                    linewidth=1.5, alpha=0.9)

        ax.set_xlabel("Year")
        ax.set_ylabel(var)
        title_scope = ("All selected countries"
                       if country == "All Countries" else country)
        ax.set_title(f"{var} · {title_scope}", pad=10)
        if len(countries) <= 18:
            ax.legend(fontsize=7, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), ncol=1,
                      borderaxespad=0)
        fig.tight_layout()
        embed_figure(fig, self.tools_plot_frame)
        self.status_var.set(
            f"Time series plotted — {var} (Innovative = blues, "
            f"Emerging = oranges)")
