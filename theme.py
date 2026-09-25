"""
theme.py — Colour palettes, ttk style application, and matplotlib defaults.

Built for the thesis interface: "Heterogeneous Impact of R&D Investment on
Economic Productivity across 25 EU Member States (1998–2023)".

Two palettes ship with the app:
  • "Palantir" — a dark analytics palette (default).
  • "Claude"   — a light palette optimised for paper-ready screenshots.

All colour values are module-level so any tab can simply do:
    import theme
    ax.set_facecolor(theme.BG)

A call to ``switch_theme(...)`` mutates these constants in-place; the next
read from any tab picks up the new values automatically.
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Default palette (Palantir dark) ─────────────────────────────────────────
BG    = "#141c27"
FG    = "#e6ecf2"
WBG   = "#1f2b3a"
WFG   = "#f2f5f9"
BLUE  = "#4ea1ff"
TEAL  = "#2dd4bf"
RED   = "#f87171"
AMBER = "#fbbf24"
GRAY  = "#7f8c8d"
DARK  = "#0d141d"
ACCENT = "#a78bfa"   # used for cluster-2 (Emerging) accents

# ── Available palettes ──────────────────────────────────────────────────────
THEMES = {
    "Palantir": dict(
        BG="#141c27", FG="#e6ecf2", WBG="#1f2b3a", WFG="#f2f5f9",
        BLUE="#4ea1ff", TEAL="#2dd4bf", RED="#f87171",
        AMBER="#fbbf24", GRAY="#7f8c8d", DARK="#0d141d",
        ACCENT="#a78bfa",
        dark_mpl=True,
    ),
    "Claude": dict(
        BG="#f7f9fc", FG="#1a2332", WBG="#ffffff", WFG="#1a2332",
        BLUE="#1f5aa8", TEAL="#137a62", RED="#b4391d",
        AMBER="#a75a13", GRAY="#6b7c8d", DARK="#e3e9f1",
        ACCENT="#6d4ec9",
        dark_mpl=False,
    ),
}
_current_theme = "Palantir"


def cluster_palette() -> list[str]:
    """Two-colour sequence for Innovative (0) vs Emerging (1) clusters."""
    return [TEAL, ACCENT, AMBER, BLUE, RED, GRAY]


def switch_theme(name: str) -> None:
    """Update every module-level colour constant to the chosen palette."""
    global BG, FG, WBG, WFG, BLUE, TEAL, RED, AMBER, GRAY, DARK, ACCENT
    global _current_theme
    t = THEMES[name]
    BG     = t["BG"];     FG     = t["FG"]
    WBG    = t["WBG"];    WFG    = t["WFG"]
    BLUE   = t["BLUE"];   TEAL   = t["TEAL"]
    RED    = t["RED"];    AMBER  = t["AMBER"]
    GRAY   = t["GRAY"];   DARK   = t["DARK"]
    ACCENT = t["ACCENT"]
    _current_theme = name
    _configure_matplotlib(t["dark_mpl"])


def _configure_matplotlib(dark: bool) -> None:
    """Apply current palette to matplotlib rcParams."""
    if dark:
        plt.style.use("dark_background")
    else:
        plt.rcdefaults()
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "semibold",
        "axes.labelsize":     10,
        "axes.labelweight":   "medium",
        "axes.facecolor":     WBG,
        "figure.facecolor":   BG,
        "savefig.facecolor":  BG,
        "text.color":         FG,
        "axes.labelcolor":    FG,
        "xtick.color":        FG,
        "ytick.color":        FG,
        "axes.edgecolor":     GRAY,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#2c3a4d" if dark else "#dbe1eb",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.alpha":         0.35,
        "legend.frameon":     False,
        "legend.fontsize":    9,
        "figure.titleweight": "bold",
        "lines.linewidth":    1.8,
        "lines.markersize":   6,
    })


def apply_theme(root) -> None:
    """Apply the current palette to every ttk widget and matplotlib."""
    from tkinter import ttk
    style = ttk.Style(root)
    style.theme_use("clam")

    base_font = ("Segoe UI", 10)
    bold_font = ("Segoe UI", 10, "bold")
    italic    = ("Segoe UI", 10, "italic")

    style.configure(".", background=BG, foreground=FG,
                    font=base_font, borderwidth=0,
                    focusthickness=2, focuscolor=BLUE)
    style.map(".", background=[("active", WBG), ("disabled", BG)],
              foreground=[("disabled", GRAY)])

    # ── Frames & labels ────────────────────────────────────────────────────
    style.configure("TFrame",      background=BG)
    style.configure("Card.TFrame", background=WBG, relief="flat")
    style.configure("TLabelframe", background=BG, foreground=FG,
                    bordercolor=BLUE, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=BG, foreground=TEAL,
                    font=("Segoe UI", 10, "bold"), padding=(4, 0))
    style.configure("TLabel", background=BG, foreground=FG)

    # Hero / title text styling
    style.configure("Title.TLabel",    background=BG, foreground=BLUE,
                    font=("Segoe UI", 18, "bold"))
    style.configure("Sub.TLabel",      background=BG, foreground=TEAL,
                    font=italic)
    style.configure("Status.TLabel",   background=BG, foreground=AMBER,
                    font=("Segoe UI", 9, "bold"))
    style.configure("Footnote.TLabel", background=BG, foreground=GRAY,
                    font=("Segoe UI", 8))

    # ── Buttons ────────────────────────────────────────────────────────────
    style.configure("TButton", background=BLUE, foreground=WFG,
                    font=bold_font, padding=(14, 7),
                    bordercolor=BLUE, relief="flat", anchor="center")
    style.map("TButton",
              background=[("active", TEAL), ("pressed", DARK),
                          ("disabled", GRAY)],
              foreground=[("active", WFG), ("pressed", WFG)],
              relief=[("pressed", "flat")])

    style.configure("Accent.TButton", background=TEAL, foreground=WFG,
                    font=bold_font, padding=(16, 8), relief="flat")
    style.map("Accent.TButton",
              background=[("active", BLUE), ("pressed", DARK)],
              foreground=[("active", WFG)])

    style.configure("Ghost.TButton", background=BG, foreground=TEAL,
                    font=bold_font, padding=(12, 6), relief="flat",
                    bordercolor=TEAL)
    style.map("Ghost.TButton",
              background=[("active", WBG), ("pressed", DARK)],
              foreground=[("active", BLUE)])

    style.configure("Theme.TButton", background=DARK, foreground=FG,
                    font=("Segoe UI", 9, "bold"), padding=(10, 5),
                    relief="flat")
    style.map("Theme.TButton",
              background=[("active", WBG), ("pressed", BG)],
              foreground=[("active", TEAL)])

    # ── Notebook / tabs ────────────────────────────────────────────────────
    style.configure("TNotebook", background=DARK, borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab",
                    background=WBG, foreground=GRAY,
                    font=("Segoe UI", 10, "bold"),
                    padding=(16, 9), borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", BG), ("active", DARK)],
              foreground=[("selected", TEAL), ("active", FG)],
              expand=[("selected", [1, 1, 1, 0])])

    # ── Entries / combos / checkboxes ──────────────────────────────────────
    style.configure("TEntry",    fieldbackground=WBG, foreground=WFG,
                    insertcolor=WFG, bordercolor=BLUE, padding=4)
    style.configure("TCombobox", fieldbackground=WBG, foreground=WFG,
                    selectbackground=BLUE, selectforeground=WFG, padding=4)
    style.map("TCombobox",
              fieldbackground=[("readonly", WBG)],
              foreground=[("readonly", WFG)])
    style.configure("TCheckbutton", background=BG, foreground=FG,
                    focuscolor=BLUE)
    style.map("TCheckbutton",
              background=[("active", BG)],
              foreground=[("active", TEAL)])
    style.configure("TRadiobutton", background=BG, foreground=FG)
    style.map("TRadiobutton",
              background=[("active", BG)],
              foreground=[("active", TEAL)])

    # ── Scroll / progress ──────────────────────────────────────────────────
    style.configure("TScrollbar",  background=WBG, troughcolor=BG,
                    arrowcolor=FG, bordercolor=BG)
    style.configure("TProgressbar", background=TEAL, troughcolor=WBG,
                    thickness=6)

    # ── Tables ─────────────────────────────────────────────────────────────
    style.configure("Treeview", background=WBG, foreground=FG,
                    fieldbackground=WBG, rowheight=24,
                    font=("Segoe UI", 9), borderwidth=0)
    style.configure("Treeview.Heading", background=DARK, foreground=TEAL,
                    font=("Segoe UI", 10, "bold"), padding=(6, 4))
    style.map("Treeview",
              background=[("selected", BLUE)],
              foreground=[("selected", WFG)])

    # ── PanedWindow separators ─────────────────────────────────────────────
    style.configure("TPanedwindow", background=DARK)
    style.configure("Sash", sashthickness=6, gripcount=0, background=DARK)

    _configure_matplotlib(dark=(_current_theme == "Palantir"))
