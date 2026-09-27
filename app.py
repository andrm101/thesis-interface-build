"""
app.py — ThesisApp: main application shell for the thesis interface.

Thesis:
    "An empirical investigation into the heterogeneous impact of
     Research & Development investment on economic productivity across
     25 EU member states over a 25-year panel (1998–2023)."

    NOTE: the shipped sample (panel_data.xlsx) covers 23 countries over
    2000-2023 -- Austria, Cyprus, and Ireland from the thesis's EU-25
    scope are not present in this shipped subsample.

The app assembles every analytical tab via multiple inheritance of mixins
(see tabs/tab_*.py).  Each mixin contributes one tab's ``_tab_XXX`` builder
and all related helper methods.  Shared state (df, ml_model, clusters, …)
lives here so every mixin can access it as ``self.XXX``.
"""

import tkinter as tk
from tkinter import ttk, messagebox

import theme
from helpers import make_text, write, clear_txt  # noqa: F401 (re-export)

# ── Tab mixins ─────────────────────────────────────────────────────────────
from tabs.tab_data        import DataTabMixin
from tabs.tab_stats       import StatsTabMixin
from tabs.tab_panel       import PanelTabMixin
from tabs.tab_convergence import ConvergenceTabMixin
from tabs.tab_clustering  import ClusterTabMixin
from tabs.tab_ml          import MLTabMixin
from tabs.tab_scenarios   import ScenariosTabMixin
from tabs.tab_compare     import CompareTabMixin
from tabs.tab_diagnostics import DiagnosticsTabMixin
from tabs.tab_unitroot    import UnitRootTabMixin
from tabs.tab_var         import VARTabMixin
from tabs.tab_advanced    import AdvancedTabMixin
from tabs.tab_report      import ReportTabMixin
from tabs.tab_causal      import CausalTabMixin


APP_TITLE    = "Augmented Solow · R&D Heterogeneity Lab"
APP_SUBTITLE = ("EU-25 sample (23 countries shipped) · 2000-2023 · World Bank DataBank   "
                "│   K-Means typology · Panel FE/RE · Hausman · Unit-Root Tests")


# Tab labels are (display label, builder attr name, refresh attr name | None).
# Display labels use a simple symbol prefix so users can find the tab at a
# glance without depending on emoji font support.
TAB_LAYOUT = [
    ("  1 · Data            ", "_tab_data",         None),
    ("  2 · Statistics      ", "_tab_stats",        "_refresh_stats_vars"),
    ("  3 · Panel FE/RE     ", "_tab_panel",        "_refresh_panel_vars"),
    ("  4 · Convergence     ", "_tab_conv",         "_refresh_conv_vars"),
    ("  5 · Clustering      ", "_tab_cluster",      "_refresh_cluster_vars"),
    ("  6 · ML Models       ", "_tab_ml",           "_refresh_ml_vars"),
    ("  7 · Scenarios       ", "_tab_scenarios",    "_refresh_sc_countries"),
    ("  8 · Compare         ", "_tab_compare",      "_refresh_compare_vars"),
    ("  9 · Diagnostics     ", "_tab_diagnostics",  "_refresh_diag_vars_tab"),
    (" 10 · Unit Roots      ", "_tab_unitroot",     "_refresh_ur_vars"),
    (" 11 · VAR / IRF       ", "_tab_var",          "_refresh_var_vars"),
    (" 12 · Advanced        ", "_tab_advanced",     "_refresh_adv_vars"),
    (" 13 · Report          ", "_tab_report",       None),
    (" 14 · Causal          ", "_tab_causal",       "_refresh_causal_vars"),
]


class ThesisApp(
    DataTabMixin,
    StatsTabMixin,
    PanelTabMixin,
    ConvergenceTabMixin,
    ClusterTabMixin,
    MLTabMixin,
    ScenariosTabMixin,
    CompareTabMixin,
    DiagnosticsTabMixin,
    UnitRootTabMixin,
    VARTabMixin,
    AdvancedTabMixin,
    ReportTabMixin,
    CausalTabMixin,
):
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "Augmented Solow · R&D Heterogeneity Lab  "
            "— EU-25 Panel Study (23 countries shipped, 2000-2023)")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry(f"{w}x{h}+0+0")
        self.root.configure(bg=theme.BG)
        theme.apply_theme(root)

        # ── Shared state ──────────────────────────────────────────────────
        self.df_raw      = None   # raw loaded DataFrame
        self.df          = None   # filtered working DataFrame
        self.ml_model    = None   # best trained ML model
        self.ml_scaler   = None
        self.ml_features: list = []
        self.ml_data     = None   # (X, y, meta, scheme, folds) of last run
        self.ml_results:  dict = {}
        self.clusters    = None   # DataFrame: Country, Cluster

        # Non-ttk widgets that need recolouring on theme switch
        self._tk_spinboxes:    list = []
        self._special_labels:  list = []  # list of (widget, colour_attr_name)

        self._build_header()
        self._build_notebook()
        self._build_footer()

    # ── Header ────────────────────────────────────────────────────────────
    def _build_header(self) -> None:
        hdr = ttk.Frame(self.root)
        hdr.pack(fill=tk.X, padx=20, pady=(14, 2))

        title_col = ttk.Frame(hdr)
        title_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_col, text=APP_TITLE,
                  style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(title_col, text=APP_SUBTITLE,
                  style="Sub.TLabel").pack(anchor=tk.W, pady=(2, 0))

        ctrl_col = ttk.Frame(hdr)
        ctrl_col.pack(side=tk.RIGHT)
        self.status_var = tk.StringVar(
            value="Load panel_data.xlsx (or any balanced panel) to begin.")
        self._status_lbl = ttk.Label(
            ctrl_col, textvariable=self.status_var, style="Status.TLabel")
        self._status_lbl.pack(side=tk.TOP, anchor=tk.E, pady=(0, 4))
        self._theme_btn = ttk.Button(ctrl_col, text="☀  Light theme  ",
                                     style="Theme.TButton",
                                     command=self._toggle_theme)
        self._theme_btn.pack(side=tk.TOP, anchor=tk.E)

        # Thin rule under the header
        sep = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=20, pady=(6, 0))

    # ── Footer ────────────────────────────────────────────────────────────
    def _build_footer(self) -> None:
        ftr = ttk.Frame(self.root)
        ftr.pack(fill=tk.X, padx=20, pady=(2, 8))
        ttk.Label(ftr,
                  text=("Augmented Solow framework  ·  GERD & patent activity  ·  "
                        "K-Means typology  ·  Panel FE / RE  ·  Hausman  ·  "
                        "ADF / PP / IPS  ·  Cross-sectional OLS"),
                  style="Footnote.TLabel").pack(side=tk.LEFT)
        ttk.Label(ftr, text="v1.1 · github.com/andrm101/thesis-interface-build",
                  style="Footnote.TLabel").pack(side=tk.RIGHT)

    # ── Theme switching ───────────────────────────────────────────────────
    def _toggle_theme(self) -> None:
        new = "Claude" if theme._current_theme == "Palantir" else "Palantir"
        theme.switch_theme(new)
        self._apply_current_theme()
        self._theme_btn.configure(
            text="☀  Light theme  " if new == "Palantir"
            else "⬛  Dark theme  ")

    def _apply_current_theme(self) -> None:
        """Re-apply colours to root, ttk styles, and every non-ttk widget."""
        self.root.configure(bg=theme.BG)
        theme.apply_theme(self.root)

        # Read-only Text widgets in each tab
        for attr in ("stats_txt", "panel_txt", "conv_txt",
                     "cluster_txt", "ml_txt", "sc_txt", "cmp_txt",
                     "diag_txt", "ur_txt", "var_txt", "adv_txt", "rpt_txt",
                     "cz_txt"):
            if hasattr(self, attr):
                getattr(self, attr).configure(
                    bg=theme.WBG, fg=theme.FG, insertbackground=theme.FG)

        # Listboxes (all of them, including the convergence control listbox)
        for attr in ("country_lb", "reg_lb", "cluster_lb", "feat_lb",
                     "conv_ctrl_lb", "cmp_reg_lb", "ur_joh_lb",
                     "var_lb", "adv_reg_lb", "rpt_model_lb", "cz_ctrl_lb"):
            if hasattr(self, attr):
                getattr(self, attr).configure(
                    bg=theme.WBG, fg=theme.FG,
                    selectbackground=theme.BLUE,
                    selectforeground=theme.WFG)

        # Non-ttk Spinboxes
        for w in self._tk_spinboxes:
            w.configure(bg=theme.WBG, fg=theme.FG,
                        insertbackground=theme.FG,
                        buttonbackground=theme.WBG,
                        disabledbackground=theme.DARK)

        # Special-coloured labels — colour attr stored as a string name
        for lbl, attr in self._special_labels:
            lbl.configure(foreground=getattr(theme, attr))

        # Status label
        if hasattr(self, "_status_lbl"):
            self._status_lbl.configure(foreground=theme.AMBER)

    # ── Notebook ──────────────────────────────────────────────────────────
    def _build_notebook(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        self._tab_refresh_map: dict[str, str] = {}
        for label, builder_name, refresh_name in TAB_LAYOUT:
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=label)
            getattr(self, builder_name)(frame)
            if refresh_name:
                # Key is the stripped label (e.g. "3 · Panel FE/RE")
                self._tab_refresh_map[label.strip()] = refresh_name
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # ── Tab-change refresh dispatch ───────────────────────────────────────
    def _on_tab_change(self, event) -> None:
        label = event.widget.tab("current", "text").strip()
        refresh_name = self._tab_refresh_map.get(label)
        if refresh_name and hasattr(self, refresh_name):
            getattr(self, refresh_name)()

    # ── Shared guard ──────────────────────────────────────────────────────
    def _check_data(self) -> bool:
        if self.df is None or self.df.empty:
            messagebox.showwarning(
                "No Data", "Load and filter a dataset first (see Tab 1).")
            return False
        return True
