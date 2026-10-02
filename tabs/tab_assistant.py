"""
tabs/tab_assistant.py  —  Tab 15 · Assistant: ask questions in plain
language; Claude answers by running the thesis's own analyses (agent/).

Every number in an answer is checked against the tool results it came from;
numbers that cannot be traced are listed as unverified. Needs the optional
`anthropic` package and an ANTHROPIC_API_KEY; without them the tab explains
what to install and stays inactive. Uses the rebuilt level panel
(data/panel_levels.csv), like the web dashboard.

Mixin: AssistantTabMixin
"""

import threading
import tkinter as tk
from tkinter import ttk

from helpers import clear_txt, make_text, write

SUGGESTIONS = (
    "Does public R&D funding crowd in private R&D?",
    "Did R&D tax reforms raise R&D intensity?",
    "How has Poland's R&D intensity changed since 2000?",
)


class AssistantTabMixin:
    # ══════════════════════════════════════════════════════════════════════
    # TAB 15 — ASSISTANT
    # ══════════════════════════════════════════════════════════════════════
    def _tab_assistant(self, parent):
        from llm import client as L
        ok, why = L.availability()
        self._as_conv = None
        self._as_busy = False

        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=12, pady=(8, 2))
        self.as_status = tk.StringVar(
            value=(f"Ready · {why}" if ok else f"Inactive · {why}"))
        ttk.Label(top, textvariable=self.as_status).pack(side=tk.LEFT)
        ttk.Button(top, text="New conversation",
                   command=self._as_reset).pack(side=tk.RIGHT)

        sug = ttk.Frame(parent)
        sug.pack(fill=tk.X, padx=12, pady=2)
        for q in SUGGESTIONS:
            ttk.Button(sug, text=q, command=lambda q=q: self._as_ask(q)
                       ).pack(side=tk.LEFT, padx=(0, 6))

        self.as_txt = make_text(parent, height=26, copy_root=self.root)
        for tag, kw in (("q", {"font": ("Consolas", 10, "bold")}),
                        ("tool", {"foreground": "#7f8c8d"}),
                        ("ok", {"foreground": "#2dd4bf"}),
                        ("warn", {"foreground": "#fbbf24"})):
            self.as_txt.tag_configure(tag, **kw)

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=12, pady=(2, 10))
        self.as_entry = ttk.Entry(row)
        self.as_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.as_entry.bind("<Return>", lambda _e: self._as_ask())
        self.as_btn = ttk.Button(row, text="Ask", command=self._as_ask)
        self.as_btn.pack(side=tk.LEFT, padx=(6, 0))
        if not ok:
            self.as_btn.state(["disabled"])
            write(self.as_txt, "The assistant needs the Claude API:\n"
                  "  pip install -r requirements-llm.txt\n"
                  "  set ANTHROPIC_API_KEY=…   (or: export ANTHROPIC_API_KEY=…)\n"
                  "then restart the app.\n", "warn")

    def _as_reset(self):
        self._as_conv = None
        clear_txt(self.as_txt)

    def _as_ask(self, question: str | None = None):
        q = (question or self.as_entry.get()).strip()
        if not q or self._as_busy or "disabled" in self.as_btn.state():
            return
        self.as_entry.delete(0, tk.END)
        self._as_busy = True
        self.as_btn.state(["disabled"])
        write(self.as_txt, f"\nYou: {q}\n", "q")
        self.status_var.set("Assistant working…")

        def on_event(kind, data):
            args = ", ".join(f"{k}={v}" for k, v in (data.get("input") or {}).items())
            self.root.after(0, lambda: write(
                self.as_txt, f"  · running {data['name']}({args})\n", "tool"))

        def work():
            from agent.runner import Assistant
            try:
                if self._as_conv is None:
                    self._as_conv = Assistant()
                reply = self._as_conv.ask(q, on_event)
                self.root.after(0, lambda: self._as_show(reply))
            except Exception as exc:  # noqa: BLE001 — shown in the tab
                self.root.after(0, lambda exc=exc: self._as_done(f"Error: {exc}\n"))
        threading.Thread(target=work, daemon=True).start()

    def _as_show(self, reply):
        write(self.as_txt, f"\n{reply.text}\n")
        g = reply.grounding
        if g.get("unverified"):
            write(self.as_txt, "  ⚠ not found in any tool result: "
                  + ", ".join(g["unverified"]) + "\n", "warn")
        elif g.get("verified"):
            write(self.as_txt, f"  ✓ all {len(g['verified'])} numbers traced to "
                  "tool results\n", "ok")
        self._as_done("")

    def _as_done(self, msg):
        if msg:
            write(self.as_txt, msg, "warn")
        self._as_busy = False
        self.as_btn.state(["!disabled"])
        self.status_var.set("Assistant ready")
