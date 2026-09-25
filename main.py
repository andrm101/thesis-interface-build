"""
main.py  —  Entry point for the app.

Run with:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

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
    ThesisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
