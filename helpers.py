"""
helpers.py  —  Shared UI helper functions used across all tabs.

All colour values are read from the `theme` module at call-time so they
reflect whichever palette is currently active.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import theme


# ── Clipboard / file helpers ───────────────────────────────────────────────
def copy_to_clipboard(txt: tk.Text, root: tk.Tk) -> None:
    """Copy the full content of a read-only Text widget to the system clipboard."""
    content = txt.get("1.0", tk.END).strip()
    root.clipboard_clear()
    root.clipboard_append(content)


def save_txt(txt: tk.Text) -> None:
    """Prompt the user for a file path and save the Text widget content."""
    path = filedialog.asksaveasfilename(
        title="Save Results",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(txt.get("1.0", tk.END))


# ── Plot embedding ─────────────────────────────────────────────────────────
def embed_figure(fig, parent, toolbar: bool = True) -> FigureCanvasTkAgg:
    """Embed a matplotlib Figure inside a tkinter parent widget."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent)
        tb.update()
    return canvas


# ── Text panel factory ─────────────────────────────────────────────────────
def make_text(parent, height: int = 20,
              copy_root: tk.Tk | None = None) -> tk.Text:
    """Create a read-only scrollable Text widget inside *parent*.

    If *copy_root* is supplied, a compact toolbar with **Copy to Clipboard**
    and **Save…** buttons is added above the text area.
    """
    outer = ttk.Frame(parent)
    outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Optional toolbar
    if copy_root is not None:
        btn_bar = ttk.Frame(outer)
        btn_bar.pack(fill=tk.X, pady=(0, 2))

    inner = ttk.Frame(outer)
    inner.pack(fill=tk.BOTH, expand=True)
    sb = ttk.Scrollbar(inner)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt = tk.Text(
        inner,
        bg=theme.WBG, fg=theme.FG,
        font=("Consolas", 10),
        insertbackground=theme.FG,
        yscrollcommand=sb.set,
        height=height,
        wrap=tk.WORD,
        relief=tk.FLAT,
        state=tk.DISABLED,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    sb.config(command=txt.yview)

    # Wire up toolbar buttons now that txt exists
    if copy_root is not None:
        ttk.Button(btn_bar, text="Copy to Clipboard",
                   command=lambda: copy_to_clipboard(txt, copy_root),
                   width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="Save…",
                   command=lambda: save_txt(txt),
                   width=8).pack(side=tk.LEFT, padx=2)

    return txt


# ── Text mutation helpers ──────────────────────────────────────────────────
def write(txt: tk.Text, msg: str, tag: str | None = None) -> None:
    """Append *msg* to a read-only Text widget, optionally with a named tag."""
    txt.configure(state=tk.NORMAL)
    if tag:
        txt.insert(tk.END, msg, tag)
    else:
        txt.insert(tk.END, msg)
    txt.see(tk.END)
    txt.configure(state=tk.DISABLED)


def clear_txt(txt: tk.Text) -> None:
    """Erase all content from a read-only Text widget."""
    txt.configure(state=tk.NORMAL)
    txt.delete("1.0", tk.END)
    txt.configure(state=tk.DISABLED)
