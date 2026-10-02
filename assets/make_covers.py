"""
make_covers.py — the project's branded banner and the wiki page covers.

    python assets/make_covers.py

Writes assets/brand-banner.svg (README header) and one cover per wiki page
in docs/wiki/images/, all in the same style: navy field, faint grid, the
teal step line and the panel-network mark.
"""
from __future__ import annotations

import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIKI_IMG = os.path.join(ROOT, "docs", "wiki", "images")

NAVY, GRID, TEAL, INK = "#081A2F", "#355C7D", "#14B8A6", "#F4F7FA"
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
NAME = "Augmented Solow · R&D Heterogeneity Lab"

# The panel-network mark (countries as nodes, teal = the R&D-led path).
MARK = f"""
    <line x1="52.5" y1="92.7" x2="89.4" y2="79.9" stroke="{GRID}" stroke-width="2.5"/>
    <line x1="52.5" y1="92.7" x2="12.6" y2="97.4" stroke="{GRID}" stroke-width="2.5"/>
    <line x1="67.7" y1="49.7" x2="89.4" y2="79.9" stroke="{GRID}" stroke-width="2.5"/>
    <line x1="107.6" y1="14.7" x2="107.2" y2="49.6" stroke="{GRID}" stroke-width="2.5"/>
    <line x1="89.4" y1="79.9" x2="107.2" y2="49.6" stroke="{GRID}" stroke-width="2.5"/>
    <line x1="42.4" y1="25.1" x2="15.8" y2="60.7" stroke="{TEAL}" stroke-width="4"/>
    <line x1="42.4" y1="25.1" x2="67.7" y2="49.7" stroke="{TEAL}" stroke-width="4"/>
    <line x1="15.8" y1="60.7" x2="12.6" y2="97.4" stroke="{TEAL}" stroke-width="4"/>
    <line x1="67.7" y1="49.7" x2="107.6" y2="14.7" stroke="{TEAL}" stroke-width="4"/>
    <rect x="34.4" y="17.1" width="16" height="16" fill="{TEAL}"/>
    <rect x="7.8" y="52.7" width="16" height="16" fill="{TEAL}"/>
    <rect x="46.5" y="86.7" width="12" height="12" fill="{NAVY}" stroke="{INK}" stroke-width="2.5"/>
    <rect x="59.7" y="41.7" width="16" height="16" fill="{TEAL}"/>
    <rect x="99.6" y="6.7" width="16" height="16" fill="{TEAL}"/>
    <rect x="83.4" y="73.9" width="12" height="12" fill="{NAVY}" stroke="{INK}" stroke-width="2.5"/>
    <rect x="4.6" y="89.4" width="16" height="16" fill="{TEAL}"/>
    <rect x="101.2" y="43.6" width="12" height="12" fill="{NAVY}" stroke="{INK}" stroke-width="2.5"/>"""


def _frame(h: int, base: int, rise: int) -> str:
    """Background, grid and the teal step line (rises to the right, kept
    below the title: *base* is its lowest level, *rise* the step height)."""
    rows = " ".join(f"M0 {y}H1280" for y in range(80, h, 80))
    steps = (f"M900 {base}H1000V{base - rise}H1100V{base - 2 * rise}H1180"
             f"V{base - rise}H1280")
    return f"""  <rect width="1280" height="{h}" fill="{NAVY}"/>
  <g stroke="{GRID}" stroke-opacity="0.22" stroke-width="1">
    <path d="{rows}"/>
    <path d="M320 0V{h}M640 0V{h}M960 0V{h}"/>
  </g>
  <g fill="none" stroke="{TEAL}" stroke-width="3" stroke-linecap="square" stroke-linejoin="miter">
    <path d="{steps}"/>
  </g>
  <g fill="{NAVY}" stroke="{TEAL}" stroke-width="3">
    <rect x="992" y="{base - 8}" width="16" height="16"/>
    <rect x="1092" y="{base - rise - 8}" width="16" height="16"/>
  </g>
  <rect x="1172" y="{base - 2 * rise - 8}" width="16" height="16" fill="{TEAL}"/>"""


def _text(x, y, s, size, weight=400, fill=INK, opacity=None, spacing=None):
    extra = (f' fill-opacity="{opacity}"' if opacity else "") + \
            (f' letter-spacing="{spacing}"' if spacing else "")
    return (f'  <text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{extra}>{escape(s)}</text>')


def banner() -> str:
    label = (f"{NAME}. Desktop app and web dashboard for thesis-grade panel "
             "analysis")
    return "\n".join([
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="320" '
         f'viewBox="0 0 1280 320" role="img" aria-label="{escape(label)}">'),
        _frame(320, 296, 40),
        f'  <g transform="translate(60 100)">{MARK}\n  </g>',
        _text(220, 140, NAME, 52, 700, spacing="-1.5"),
        _text(222, 182, "Desktop app and web dashboard for thesis-grade panel analysis",
              24, 500, TEAL),
        _text(222, 222, "EU-25 · 1998–2023 · panel FE, GMM, convergence, causal designs",
              20, opacity="0.8"),
        _text(222, 258, "Python · Tkinter · FastAPI · Angular · ECharts", 16,
              opacity="0.55"),
        "</svg>", ""])


def cover(title: str, subtitle: str) -> str:
    return "\n".join([
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="240" '
         f'viewBox="0 0 1280 240" role="img" aria-label="{escape(title)} — '
         f'{escape(subtitle)}">'),
        _frame(240, 212, 44),
        f'  <g transform="translate(60 62)">{MARK}\n  </g>',
        _text(222, 78, NAME.upper(), 14, 600, TEAL, spacing="2"),
        _text(220, 132, title, 48, 700, spacing="-1.2"),
        _text(222, 172, subtitle, 22, 400, opacity="0.8"),
        "</svg>", ""])


PAGES = {
    "home": ("Companion wiki", "Thesis results, methods, data and the web dashboard"),
    "key-findings": ("Key Findings", "What the evidence says about R&D, convergence and policy"),
    "methodology": ("Methodology", "From the raw workbook to dynamic panels and causal designs"),
    "data-and-audit": ("Data and Audit", "Sources, corrections and policy-data coverage"),
    "reproducing-results": ("Reproducing Results", "Every number, figure and table from one command"),
    "web-dashboard": ("Web Dashboard", "Explore the panel and rerun the analyses in a browser"),
    "limitations": ("Limitations and Next Steps", "What the evidence cannot say yet"),
}


def main() -> None:
    with open(os.path.join(HERE, "brand-banner.svg"), "w", encoding="utf-8") as fh:
        fh.write(banner())
    os.makedirs(WIKI_IMG, exist_ok=True)
    for slug, (title, sub) in PAGES.items():
        with open(os.path.join(WIKI_IMG, f"cover-{slug}.svg"), "w",
                  encoding="utf-8") as fh:
            fh.write(cover(title, sub))
    print(f"wrote assets/brand-banner.svg and {len(PAGES)} wiki covers")


if __name__ == "__main__":
    main()
