"""
tabs/tab_report.py  —  Report & Export tab.

Features
────────
  · Model Registry  — capture Panel OLS / VAR / Advanced results by name
  · LaTeX Table     — stargazer-style multi-column regression table (.tex)
  · HTML Report     — full auto-generated report with embedded plots + tables
  · Data Export     — filtered DataFrame → CSV or Excel (multi-sheet)
  · Figure Export   — batch export all current plots as PNG into a folder
"""

import io
import os
import base64
import datetime
import threading
import html as html_lib
import zipfile
import webbrowser
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

import theme
from helpers import make_text, write, clear_txt


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _stars_tex(p):
    return ("^{***}" if p < 0.01 else "^{**}" if p < 0.05
            else "^{*}" if p < 0.10 else "")


def _stars_str(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _extract(result):
    """
    Unified extractor for linearmodels and statsmodels results.
    Returns (params, ses, pvalues, nobs, rsquared, model_label).
    """
    try:
        # linearmodels (PanelOLS, RandomEffects, PooledOLS, IV2SLS …)
        params = result.params
        ses    = result.std_errors
        pvals  = result.pvalues
        nobs   = int(result.nobs)
        try:
            rsq = float(result.rsquared)
        except Exception:
            rsq = float("nan")
        label = type(result).__name__.replace("Results", "")
        return params, ses, pvals, nobs, rsq, label
    except AttributeError:
        pass
    # statsmodels OLS / QuantReg / etc.
    params = result.params
    ses    = result.bse
    pvals  = result.pvalues
    nobs   = int(result.nobs)
    rsq    = float(getattr(result, "rsquared", float("nan")))
    label  = type(result).__name__.replace("RegressionResults", "OLS")
    return params, ses, pvals, nobs, rsq, label


def _fig_to_b64(fig, dpi=110):
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═════════════════════════════════════════════════════════════════════════════
# Mixin
# ═════════════════════════════════════════════════════════════════════════════

class ReportTabMixin:

    # ── Tab builder ───────────────────────────────────────────────────────────
    def _tab_report(self, parent):
        if not hasattr(self, "_report_models"):
            self._report_models = []          # [(name, result_obj), …]
        if not hasattr(self, "_report_figs"):
            self._report_figs   = {}          # {name: fig} for batch export

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left  = ttk.Frame(pane); pane.add(left,  weight=2)
        right = ttk.Frame(pane); pane.add(right, weight=3)

        self._build_report_controls(left)

        # Preview text (right)
        ttk.Label(right, text="Preview / Output",
                  style="Sub.TLabel").pack(anchor=tk.W, padx=4)
        self.rpt_txt = make_text(right, height=40, copy_root=self.root)

    # ── Controls (left panel) ─────────────────────────────────────────────────
    def _build_report_controls(self, parent):
        # ── Model Registry ────────────────────────────────────────────────────
        rf = ttk.LabelFrame(parent, text=" Model Registry")
        rf.pack(fill=tk.X, padx=8, pady=(6, 4))

        self.rpt_model_lb = tk.Listbox(
            rf, height=5, width=36,
            bg=theme.WBG, fg=theme.FG,
            selectbackground=theme.BLUE,
            selectforeground=theme.WFG,
            exportselection=False)
        self.rpt_model_lb.pack(fill=tk.X, padx=4, pady=2)
        self._tk_spinboxes   # keep track hook already exists

        capture_row = ttk.Frame(rf)
        capture_row.pack(fill=tk.X, padx=4, pady=2)
        for label, cmd in [
            ("+ Panel OLS",  self._capture_panel),
            ("+ VAR",        self._capture_var),
            ("+ Advanced",   self._capture_adv),
            ("✕ Remove",     self._remove_model),
        ]:
            ttk.Button(capture_row, text=label, width=10,
                       command=cmd).pack(side=tk.LEFT, padx=2)

        # ── LaTeX Regression Table ────────────────────────────────────────────
        ltf = ttk.LabelFrame(parent, text=" LaTeX Regression Table")
        ltf.pack(fill=tk.X, padx=8, pady=4)

        opts = ttk.Frame(ltf)
        opts.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(opts, text="SE format:").pack(side=tk.LEFT)
        self.rpt_se_fmt = ttk.Combobox(
            opts, width=16, state="readonly",
            values=["Parentheses", "Brackets", "None"])
        self.rpt_se_fmt.current(0)
        self.rpt_se_fmt.pack(side=tk.LEFT, padx=4)

        self.rpt_notes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Notes row",
                        variable=self.rpt_notes_var).pack(side=tk.LEFT, padx=6)

        btn_row = ttk.Frame(ltf)
        btn_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        for label, cmd in [
            ("Preview LaTeX",   self._preview_latex),
            ("Save .tex file",  self._save_latex),
            ("Copy to Clipboard", self._copy_latex),
        ]:
            ttk.Button(btn_row, text=label, command=cmd).pack(
                side=tk.LEFT, padx=2)

        # ── HTML Report ───────────────────────────────────────────────────────
        htf = ttk.LabelFrame(parent, text=" HTML Report")
        htf.pack(fill=tk.X, padx=8, pady=4)

        sect_grid = ttk.Frame(htf)
        sect_grid.pack(fill=tk.X, padx=4, pady=2)
        self._html_sections = {}
        sections = [
            ("Summary Stats",    "stats"),
            ("Panel OLS",        "panel"),
            ("Convergence",      "conv"),
            ("Clustering",       "cluster"),
            ("ML Models",        "ml"),
            ("Diagnostics",      "diag"),
            ("Unit Roots",       "ur"),
            ("VAR / IRF",        "var"),
            ("Advanced",         "adv"),
        ]
        for idx, (label, key) in enumerate(sections):
            v = tk.BooleanVar(value=True)
            self._html_sections[key] = v
            ttk.Checkbutton(sect_grid, text=label, variable=v).grid(
                row=idx // 3, column=idx % 3,
                sticky=tk.W, padx=6, pady=1)

        html_btns = ttk.Frame(htf)
        html_btns.pack(fill=tk.X, padx=4, pady=(2, 6))
        ttk.Button(html_btns, text="Generate & Open in Browser",
                   style="Accent.TButton",
                   command=self._generate_html_report).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(html_btns, text="Save HTML",
                   command=self._save_html_report).pack(side=tk.LEFT, padx=2)

        # ── Data Export ───────────────────────────────────────────────────────
        dtf = ttk.LabelFrame(parent, text=" Data Export")
        dtf.pack(fill=tk.X, padx=8, pady=4)
        data_row = ttk.Frame(dtf)
        data_row.pack(fill=tk.X, padx=4, pady=4)
        for label, cmd in [
            ("Export CSV",        self._export_csv),
            ("Export Excel",      self._export_excel),
        ]:
            ttk.Button(data_row, text=label,
                       command=cmd).pack(side=tk.LEFT, padx=4)

        # ── Figure Export ─────────────────────────────────────────────────────
        ftf = ttk.LabelFrame(parent, text=" Figure Export")
        ftf.pack(fill=tk.X, padx=8, pady=4)
        fig_row = ttk.Frame(ftf)
        fig_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(fig_row, text="Export Figures to Folder",
                   command=self._export_figures).pack(side=tk.LEFT, padx=4)
        ttk.Label(fig_row, text="(saves fresh summary plots as PNG)",
                  foreground=theme.GRAY).pack(side=tk.LEFT, padx=6)

    # ═════════════════════════════════════════════════════════════════════════
    # Model Registry
    # ═════════════════════════════════════════════════════════════════════════
    def _capture_model(self, result_attr, default_prefix):
        result = getattr(self, result_attr, None)
        if result is None:
            messagebox.showwarning(
                "No result",
                f"Run a {default_prefix} first, then capture it.")
            return
        name = simpledialog.askstring(
            "Model Name",
            f"Label for this {default_prefix} result:",
            initialvalue=f"{default_prefix} {len(self._report_models)+1}")
        if not name:
            return
        self._report_models.append((name, result))
        self.rpt_model_lb.insert(tk.END, name)

    def _capture_panel(self): self._capture_model("panel_result", "Panel OLS")
    def _capture_var(self):   self._capture_model("var_result",   "VAR")

    def _capture_adv(self):
        """Capture the last OLS result stored from the Advanced tab."""
        result = getattr(self, "_adv_last_result", None)
        if result is None:
            messagebox.showwarning(
                "No result",
                "Run Driscoll-Kraay or another Advanced method first.")
            return
        name = simpledialog.askstring(
            "Model Name", "Label for this Advanced result:",
            initialvalue=f"Advanced {len(self._report_models)+1}")
        if not name:
            return
        self._report_models.append((name, result))
        self.rpt_model_lb.insert(tk.END, name)

    def _remove_model(self):
        sel = self.rpt_model_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        self.rpt_model_lb.delete(idx)
        del self._report_models[idx]

    # ═════════════════════════════════════════════════════════════════════════
    # LaTeX Table
    # ═════════════════════════════════════════════════════════════════════════
    def _build_latex_table(self):
        if not self._report_models:
            raise ValueError(
                "No models in registry.\n"
                "Use the '+ Panel OLS / + VAR / + Advanced' buttons to add results.")

        models   = self._report_models
        se_fmt   = self.rpt_se_fmt.get()
        add_notes = self.rpt_notes_var.get()

        # Collect all variable names (exclude const / Intercept / FE dummies)
        _skip = {"const", "Intercept", "intercept"}
        all_vars = []
        model_data = []
        for name, res in models:
            params, ses, pvals, nobs, rsq, lbl = _extract(res)
            model_data.append((name, params, ses, pvals, nobs, rsq))
            for v in params.index:
                if v not in _skip and not v.startswith("FE:") and v not in all_vars:
                    all_vars.append(v)

        ncols = len(models)
        col_spec = "l" + "c" * ncols
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Regression Results}",
            r"\label{tab:regression_results}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            " & " + " & ".join(
                f"({i+1})" for i in range(ncols)) + r" \\",
            " & " + " & ".join(
                name for name, *_ in model_data) + r" \\",
            r"\midrule",
        ]

        def _fmt_open(s):
            return {"Parentheses": f"({s})",
                    "Brackets":    f"[{s}]",
                    "None":        s}.get(se_fmt, f"({s})")

        for var in all_vars:
            coef_cells = []
            se_cells   = []
            for _, params, ses, pvals, _, _ in model_data:
                if var in params.index:
                    c = params[var]
                    s = ses[var]
                    p = pvals[var]
                    coef_cells.append(f"{c:.4f}{_stars_tex(p)}")
                    se_cells.append(_fmt_open(f"{s:.4f}"))
                else:
                    coef_cells.append("")
                    se_cells.append("")
            # Escape _ in variable names
            var_tex = var.replace("_", r"\_")
            lines.append(var_tex + " & " +
                         " & ".join(coef_cells) + r" \\")
            if se_fmt != "None":
                lines.append(" & " + " & ".join(se_cells) + r" \\")
            lines.append("")

        lines.append(r"\midrule")
        # Bottom stats
        nobs_cells = [str(nobs) for *_, nobs, _ in model_data]
        rsq_cells  = [f"{rsq:.4f}" if not np.isnan(rsq) else ""
                      for *_, nobs, rsq in model_data]
        lines.append(r"$N$ & " + " & ".join(nobs_cells) + r" \\")
        lines.append(r"$R^{2}$ & " + " & ".join(rsq_cells) + r" \\")

        lines.append(r"\bottomrule")
        if add_notes:
            fn = (r"\multicolumn{" + str(ncols + 1) + r"}{l}"
                  r"{\footnotesize \textit{Standard errors in parentheses. "
                  r"$^{***}$~p$<$0.01, $^{**}$~p$<$0.05, $^{*}$~p$<$0.10.}}"
                  r" \\")
            lines.append(fn)
        lines += [r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def _preview_latex(self):
        try:
            tex = self._build_latex_table()
            clear_txt(self.rpt_txt)
            write(self.rpt_txt, tex + "\n")
        except Exception as exc:
            messagebox.showerror("LaTeX Error", str(exc))

    def _save_latex(self):
        try:
            tex = self._build_latex_table()
        except Exception as exc:
            messagebox.showerror("LaTeX Error", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="Save LaTeX Table",
            defaultextension=".tex",
            filetypes=[("TeX files", "*.tex"), ("All files", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(tex)
            messagebox.showinfo("Saved", f"LaTeX table saved to:\n{path}")

    def _copy_latex(self):
        try:
            tex = self._build_latex_table()
        except Exception as exc:
            messagebox.showerror("LaTeX Error", str(exc))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(tex)
        messagebox.showinfo("Copied", "LaTeX table copied to clipboard.")

    # ═════════════════════════════════════════════════════════════════════════
    # HTML Report
    # ═════════════════════════════════════════════════════════════════════════
    _HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Thesis Report — Extended Kasim Model</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif;
         max-width: 960px; margin: 0 auto; padding: 30px;
         background: #fafafa; color: #222; }}
  h1   {{ border-bottom: 3px solid #2563eb; padding-bottom: 8px; color: #1e3a5f; }}
  h2   {{ border-bottom: 1px solid #aaa; color: #1e3a5f; margin-top: 36px; }}
  h3   {{ color: #444; }}
  pre  {{ background: #f0f4f8; padding: 14px; border-radius: 6px;
         overflow-x: auto; font-size: 12px; line-height: 1.5;
         white-space: pre-wrap; word-break: break-all; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0;
           font-size: 13px; }}
  th   {{ background: #2563eb; color: white; padding: 6px 10px; text-align: center; }}
  td   {{ border: 1px solid #ddd; padding: 5px 10px; text-align: right; }}
  td:first-child {{ text-align: left; }}
  tr:nth-child(even) {{ background: #f5f8ff; }}
  img  {{ max-width: 100%; border: 1px solid #ddd;
          border-radius: 4px; margin: 10px 0; }}
  .meta   {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .badge  {{ display: inline-block; background: #dbeafe;
             color: #1e40af; border-radius: 4px;
             padding: 2px 8px; font-size: 12px; margin: 2px; }}
  .section {{ margin-top: 40px; }}
  footer   {{ margin-top: 60px; color: #999; font-size: 11px;
              border-top: 1px solid #eee; padding-top: 10px; }}
</style>
</head>
<body>
"""
    _HTML_FOOT = """
<footer>
  Generated by <strong>Extended Kasim Model Interface</strong> &mdash;
  {date}
</footer>
</body></html>
"""

    def _html_section(self, key, title, txt_attr, plot_gen_fn=None):
        """Build one HTML section from a text widget + optional fresh plot."""
        if not self._html_sections.get(key, tk.BooleanVar(value=False)).get():
            return ""
        content = ""
        if hasattr(self, txt_attr):
            raw = getattr(self, txt_attr).get("1.0", tk.END).strip()
            if raw:
                content += f"<pre>{html_lib.escape(raw)}</pre>\n"
        if plot_gen_fn:
            try:
                fig = plot_gen_fn()
                if fig:
                    b64 = _fig_to_b64(fig)
                    content += f'<img src="data:image/png;base64,{b64}" />\n'
                    plt.close(fig)
            except Exception:
                pass
        if not content:
            return ""
        return (f'<div class="section">\n'
                f'<h2>{html_lib.escape(title)}</h2>\n'
                f'{content}\n</div>\n')

    # ── Fresh summary plots for HTML report ───────────────────────────────────
    def _html_fig_summary_stats(self):
        if self.df is None:
            return None
        num = self.df.select_dtypes("number").drop(columns=["Year"],
                                                    errors="ignore")
        if num.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        corr = num.corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        for (i, j), v in np.ndenumerate(corr.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("Correlation Matrix", fontsize=11)
        fig.tight_layout()
        return fig

    def _html_fig_panel_coefs(self):
        res = getattr(self, "panel_result", None)
        if res is None:
            return None
        try:
            params, ses, pvals, *_ = _extract(res)
            skip = {"const", "Intercept", "intercept"}
            vars_ = [v for v in params.index
                     if v not in skip and not v.startswith("FE:")]
            coefs = params[vars_].values
            errs  = ses[vars_].values
            y_pos = np.arange(len(vars_))
            fig, ax = plt.subplots(
                figsize=(7, max(3, len(vars_) * 0.45 + 1)),
                facecolor="white")
            colors = ["#2563eb" if c > 0 else "#dc2626" for c in coefs]
            ax.barh(y_pos, coefs, color=colors, height=0.5, alpha=0.8)
            ax.errorbar(coefs, y_pos, xerr=1.96 * errs,
                        fmt="none", ecolor="#555", capsize=4, linewidth=1.2)
            ax.axvline(0, color="#999", linewidth=0.8, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(vars_, fontsize=8)
            ax.set_xlabel("Coefficient  (± 1.96 SE)")
            ax.set_title("Panel OLS Coefficients", fontsize=11)
            fig.tight_layout()
            return fig
        except Exception:
            return None

    def _html_fig_ml_importance(self):
        if not self.ml_results or not self.ml_features:
            return None
        try:
            model = self.ml_model
            if not hasattr(model, "feature_importances_"):
                return None
            imp  = model.feature_importances_
            feat = self.ml_features
            idx  = np.argsort(imp)[::-1]
            fig, ax = plt.subplots(figsize=(7, max(3, len(feat)*0.4+1)),
                                    facecolor="white")
            ax.barh(range(len(feat)), imp[idx], color="#2563eb", alpha=0.8)
            ax.set_yticks(range(len(feat)))
            ax.set_yticklabels([feat[i] for i in idx], fontsize=8)
            ax.set_xlabel("Feature Importance")
            ax.set_title("ML Model — Feature Importance", fontsize=11)
            fig.tight_layout()
            return fig
        except Exception:
            return None

    def _html_fig_cluster(self):
        if self.clusters is None:
            return None
        try:
            df_c = self.clusters
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="white")
            counts = df_c["Cluster"].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color="#2563eb",
                   alpha=0.8)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of Countries")
            ax.set_title("Country Cluster Distribution", fontsize=11)
            fig.tight_layout()
            return fig
        except Exception:
            return None

    def _build_html(self):
        if self.df is None:
            raise ValueError("Load a dataset first.")

        now  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
        n    = len(self.df)
        countries = sorted(self.df["Country"].unique()) if "Country" in self.df else []
        years     = sorted(self.df["Year"].unique()) if "Year" in self.df else []
        num_cols  = [c for c in self.df.select_dtypes("number").columns
                     if c != "Year"]

        # Summary stats table (HTML)
        desc = self.df[num_cols].describe().round(4)
        stats_html  = "<table>\n<thead><tr><th>Variable</th>"
        stats_html += "".join(f"<th>{c}</th>" for c in desc.columns)
        stats_html += "</tr></thead>\n<tbody>\n"
        for idx_name, row in desc.iterrows():
            stats_html += (f"<tr><td>{html_lib.escape(str(idx_name))}</td>"
                           + "".join(f"<td>{v:.4g}</td>" for v in row)
                           + "</tr>\n")
        stats_html += "</tbody></table>\n"

        parts = [
            self._HTML_HEAD,
            "<h1>Extended Kasim Model of Endogenous Growth</h1>",
            f'<p class="meta">',
            f"  Generated: <strong>{now}</strong> &nbsp;|&nbsp; "
            f"  Observations: <strong>{n:,}</strong> &nbsp;|&nbsp; "
            f"  Variables: <strong>{len(num_cols)}</strong>",
            f"</p>",
            f'<p>{"".join(f"<span class=badge>{html_lib.escape(c)}</span>" for c in countries[:20])}'
            + ("…" if len(countries) > 20 else "") + "</p>",
            f"<p>Years: {min(years) if years else '—'} – {max(years) if years else '—'}</p>",
        ]

        # Dataset summary section (always included)
        parts += [
            '<div class="section"><h2>Dataset Summary</h2>',
            stats_html,
        ]
        fig_ss = self._html_fig_summary_stats()
        if fig_ss:
            parts.append(
                f'<img src="data:image/png;base64,{_fig_to_b64(fig_ss)}" />')
            plt.close(fig_ss)
        parts.append("</div>")

        # Panel OLS section
        parts.append(
            self._html_section("panel", "Panel OLS Results", "panel_txt",
                               self._html_fig_panel_coefs))

        # Other text sections
        for key, title, attr in [
            ("stats",   "Summary Statistics",  "stats_txt"),
            ("conv",    "Convergence Analysis","conv_txt"),
            ("cluster", "Cluster Analysis",    "cluster_txt"),
            ("ml",      "Machine Learning",    "ml_txt"),
            ("diag",    "Regression Diagnostics","diag_txt"),
            ("ur",      "Unit Root Tests",     "ur_txt"),
            ("var",     "VAR / IRF Analysis",  "var_txt"),
            ("adv",     "Advanced Econometrics","adv_txt"),
        ]:
            extra_fn = None
            if key == "cluster":
                extra_fn = self._html_fig_cluster
            elif key == "ml":
                extra_fn = self._html_fig_ml_importance
            parts.append(self._html_section(key, title, attr, extra_fn))

        # LaTeX table appendix (if models are captured)
        if self._report_models:
            try:
                tex = self._build_latex_table()
                parts += [
                    '<div class="section">',
                    "<h2>LaTeX Regression Table</h2>",
                    f"<pre>{html_lib.escape(tex)}</pre>",
                    "</div>",
                ]
            except Exception:
                pass

        parts.append(self._HTML_FOOT.format(date=now))
        return "\n".join(parts)

    def _generate_html_report(self):
        def _work():
            try:
                html_str = self._build_html()
                # Write to temp file and open browser
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".html",
                    mode="w", encoding="utf-8")
                tmp.write(html_str)
                tmp.close()
                self.root.after(0, lambda: webbrowser.open(
                    "file://" + tmp.name.replace("\\", "/")))
                # Show preview (first 3000 chars)
                preview = html_str[:3000] + "\n\n… (truncated — see browser)"
                self.root.after(0, lambda: (clear_txt(self.rpt_txt),
                                            write(self.rpt_txt, preview)))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "HTML Report Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    def _save_html_report(self):
        def _work():
            try:
                html_str = self._build_html()
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "HTML Error", str(exc)))
                return
            # Tk dialogs must run on the main thread.
            self.root.after(0, lambda: _ask_and_save(html_str))

        def _ask_and_save(html_str):
            path = filedialog.asksaveasfilename(
                title="Save HTML Report",
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")])
            if path:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(html_str)
                messagebox.showinfo("Saved", f"Report saved to:\n{path}")

        threading.Thread(target=_work, daemon=True).start()

    # ═════════════════════════════════════════════════════════════════════════
    # Data Export
    # ═════════════════════════════════════════════════════════════════════════
    def _export_csv(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.df.to_csv(path, index=False, encoding="utf-8")
            clear_txt(self.rpt_txt)
            write(self.rpt_txt,
                  f"✓ CSV exported:\n  {path}\n\n"
                  f"  Rows: {len(self.df):,}\n"
                  f"  Columns: {len(self.df.columns)}\n"
                  f"  Countries: {self.df['Country'].nunique() if 'Country' in self.df else '—'}\n"
                  f"  Years: {self.df['Year'].min() if 'Year' in self.df else '—'}"
                  f" – {self.df['Year'].max() if 'Year' in self.df else '—'}\n")

    def _export_excel(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return

        def _work():
            try:
                num_cols = [c for c in self.df.select_dtypes("number").columns
                            if c != "Year"]
                desc = self.df[num_cols].describe().round(6)

                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    # Sheet 1: full data
                    self.df.to_excel(writer, sheet_name="Data",
                                     index=False)
                    # Sheet 2: summary stats
                    desc.to_excel(writer, sheet_name="Summary Stats")
                    # Sheet 3: correlation matrix
                    self.df[num_cols].corr().round(4).to_excel(
                        writer, sheet_name="Correlations")
                    # Sheet 4: per-country averages
                    if "Country" in self.df.columns:
                        self.df.groupby("Country")[num_cols].mean().round(4).to_excel(
                            writer, sheet_name="Country Averages")
                    # Sheet 5: per-year averages
                    if "Year" in self.df.columns:
                        self.df.groupby("Year")[num_cols].mean().round(4).to_excel(
                            writer, sheet_name="Year Averages")

                msg = (f"✓ Excel exported:\n  {path}\n\n"
                       f"  Sheets: Data, Summary Stats, Correlations, "
                       f"Country Averages, Year Averages\n"
                       f"  Rows: {len(self.df):,}\n")
                self.root.after(0, lambda: (clear_txt(self.rpt_txt),
                                            write(self.rpt_txt, msg)))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Exported", f"Excel file saved to:\n{path}"))
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror(
                    "Excel Export Error", str(exc)))

        threading.Thread(target=_work, daemon=True).start()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure Export
    # ═════════════════════════════════════════════════════════════════════════
    def _export_figures(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        folder = filedialog.askdirectory(title="Select folder for figures")
        if not folder:
            return

        def _work():
            saved  = []
            errors = []

            # Map: filename → figure-generator function
            generators = {
                "01_correlation_matrix":  self._html_fig_summary_stats,
                "02_panel_ols_coefs":     self._html_fig_panel_coefs,
                "03_ml_importance":       self._html_fig_ml_importance,
                "04_cluster_distribution":self._html_fig_cluster,
            }
            # Also try to regenerate key distribution plots
            try:
                num_cols = [c for c in self.df.select_dtypes("number").columns
                            if c != "Year"][:6]
                if num_cols:
                    fig, axes = plt.subplots(
                        2, max(1, len(num_cols) // 2 + 1),
                        figsize=(12, 6), facecolor="white")
                    for idx, col in enumerate(num_cols):
                        ax = axes.flat[idx] if hasattr(axes, "flat") else axes
                        ax.hist(self.df[col].dropna(), bins=25,
                                color="#2563eb", alpha=0.8, edgecolor="white")
                        ax.set_title(col, fontsize=9)
                    for extra_ax in axes.flat[len(num_cols):]:
                        extra_ax.axis("off")
                    fig.suptitle("Variable Distributions", fontsize=12)
                    fig.tight_layout()
                    generators["05_distributions"] = lambda: fig
            except Exception:
                pass

            for fname, gen_fn in generators.items():
                try:
                    fig = gen_fn()
                    if fig is None:
                        continue
                    out_path = os.path.join(folder, f"{fname}.png")
                    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                                facecolor="white")
                    plt.close(fig)
                    saved.append(out_path)
                except Exception as e:
                    errors.append(f"{fname}: {e}")

            msg = f"✓ Figures exported to:\n  {folder}\n\n"
            msg += f"  Saved ({len(saved)}):\n"
            for p in saved:
                msg += f"    {os.path.basename(p)}\n"
            if errors:
                msg += f"\n  Errors ({len(errors)}):\n"
                for e in errors:
                    msg += f"    {e}\n"

            self.root.after(0, lambda: (clear_txt(self.rpt_txt),
                                        write(self.rpt_txt, msg)))

        threading.Thread(target=_work, daemon=True).start()
