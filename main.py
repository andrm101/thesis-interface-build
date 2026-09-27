"""
main.py  —  Entry point for the app.

Run with:
    python main.py                 # auto-loads the bundled panel_data.xlsx
    python main.py path/to/panel   # load another .xlsx / .csv panel
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import tkinter as tk
from app import ThesisApp


def main() -> None:
    # Enable per-monitor DPI awareness on Windows so the UI is sharp on
    # high-resolution displays.
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    app = ThesisApp(root)

    # Pre-load a dataset: an explicit CLI path, else the bundled sample.
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "panel_data.xlsx")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if os.path.exists(path):
        root.after(100, lambda: app.load_path(path))

    root.mainloop()


if __name__ == "__main__":
    main()
