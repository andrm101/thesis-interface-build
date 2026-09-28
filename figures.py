"""
figures.py — Publication figures and LaTeX tables for the thesis write-up.

    python reproduce.py --figures        # results + results/figures + results/tables

Style: print/paper, white background, one y-axis per panel, thin marks,
recessive grid. Colours are the first three slots of a validated
categorical palette (all-pairs CVD ΔE ≥ 9, normal-vision ΔE ≥ 24); identity
never rests on colour alone — every series also has its own marker shape
and a legend or direct label, so figures survive greyscale printing.
Group colours are fixed by entity: Innovative = slot 1, Emerging = slot 2.
"""

from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GROUP_STYLE = {"Innovative": (BLUE, "o"), "Emerging": (ORANGE, "s"),
               "All": (AQUA, "D")}
WIDTH = 6.3                                   # inches, A4 text width
# No creation timestamps, so regenerated figures are byte-identical when
# their content is unchanged (clean git diffs).
_META = {"pdf": {"CreationDate": None, "ModDate": None},
         "png": {"Software": None}}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.6,
        "lines.markersize": 5.5, "legend.frameon": False,
        "legend.fontsize": 8, "font.family": "DejaVu Sans"})


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight", pad_inches=0.05,
                    metadata=_META[ext])
    plt.close(fig)
    return name


def _legend_below(ax, ncol=2, y=-0.16):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def _label_points(ax, xs, ys, names, fontsize=6.5):
    """Greedy collision-avoiding point labels: try eight offsets around
    each point, keep the first whose box overlaps no placed label."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    placed = []
    offs = [(4, 3, "left", "bottom"), (4, -3, "left", "top"),
            (-4, 3, "right", "bottom"), (-4, -3, "right", "top"),
            (0, 6, "center", "bottom"), (0, -6, "center", "top"),
            (8, 0, "left", "center"), (-8, 0, "right", "center")]
    for x, y, n in sorted(zip(xs, ys, names), key=lambda t: -t[1]):
        for dx, dy, ha, va in offs:
            t = ax.annotate(n, (x, y), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK2)
            bb = t.get_window_extent(r).expanded(1.05, 1.1)
            if not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()
        else:   # nothing free: accept the first slot
            dx, dy, ha, va = offs[0]
            ax.annotate(n, (x, y), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=fontsize, color=INK2)


def _zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK2, lw=0.8)


# ── LaTeX tables ───────────────────────────────────────────────────────────
def to_latex(df: pd.DataFrame, caption: str, label: str,
             floatfmt: str = "{:.3f}") -> str:
    """booktabs table without the jinja2 dependency of DataFrame.to_latex."""
    def esc(x):
        t = (str(x).replace("\\", r"\textbackslash{}").replace("&", r"\&")
             .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#"))
        t = re.sub(r"\*+", lambda m: f"$^{{{m.group(0)}}}$", t)
        for a_, b_ in (("★", r"$\star$"), ("✓", r"\checkmark"), ("✗", "--"),
                       ("θ", r"$\theta$"), ("β", r"$\beta$"),
                       ("∂", r"$\partial$"), ("Δ", r"$\Delta$"),
                       ("→", r"$\to$"), ("×", r"$\times$"),
                       ("λ", r"$\lambda$"), ("W̄", r"$\bar{W}$"),
                       ("Z̃", r"$\tilde{Z}$"), ("—", "--"), ("²", r"$^2$")):
            t = t.replace(a_, b_)
        return t

    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    out = [r"\begin{table}[htbp]", r"\centering\small",
           rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(esc(v))
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\par\footnotesize Significance: $^{*}$ 10\%, $^{**}$ 5\%, "
            r"$^{***}$ 1\%.", r"\end{table}", ""]
    return "\n".join(out)


# ── figures ────────────────────────────────────────────────────────────────
def fig_typology(df, cols, groups, outdir):
    cm = df.groupby("Country")[cols].mean().dropna()
    X = StandardScaler().fit_transform(cm)
    pca = PCA(2).fit(X)
    P = pca.transform(X)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    for gname in ("Innovative", "Emerging"):
        m = (groups.reindex(cm.index) == gname).to_numpy()
        col, mk = GROUP_STYLE[gname]
        ax.scatter(P[m, 0], P[m, 1], c=col, marker=mk, s=42,
                   edgecolor="white", linewidth=0.8, label=gname, zorder=3)
    _label_points(ax, P[:, 0], P[:, 1], list(cm.index))
    ve = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ve[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC2 ({ve[1]:.0f} % of variance)")
    ax.set_title("R&D typology (PCA of country-mean R&D levels)",
                 loc="left")
    ax.legend(loc="best")
    return _save(fig, outdir, "fig1_typology_pca")


def fig_clubs(logx, res, outdir):
    from clubs import transition_paths
    h = transition_paths(logx)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    styles = [(BLUE, "o"), (ORANGE, "s"), (AQUA, "D")]
    groups = [(f"Club {k}", m) for k, m in enumerate(res.clubs, 1)]
    for i, (name, m) in enumerate(groups):
        col, mk = styles[i] if i < 3 else (INK2, "^")
        path = h[m].mean(axis=1)
        ax.plot(path.index, path, color=col, marker=mk, markevery=4,
                label=f"{name} (n = {len(m)})")
        ax.annotate(name, (path.index[-1], path.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK)
    if res.divergent:
        path = h[res.divergent].mean(axis=1)
        ax.plot(path.index, path, color=GREY, ls="--", marker="x",
                markevery=4, label=f"Non-convergent (n = {len(res.divergent)})")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_ylabel("relative transition path $h_{it}$")
    ax.set_title("Phillips-Sul convergence clubs, log output per worker",
                 loc="left")
    _legend_below(ax, ncol=2, y=-0.12)
    ax.margins(x=0.08)
    return _save(fig, outdir, "fig2_convergence_clubs")


def fig_jcurve(profiles, outdir):
    """profiles: {track label: DataFrame(Sample, L, coef, se)}."""
    fig, axes = plt.subplots(1, len(profiles), figsize=(WIDTH, 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, t) in zip(axes, profiles.items()):
        for j, gname in enumerate(("Innovative", "Emerging")):
            s = t[t.Sample == gname]
            col, mk = GROUP_STYLE[gname]
            x = s.L + (j - 0.5) * 0.14
            ax.errorbar(x, s.coef, yerr=1.96 * s.se, color=col, marker=mk,
                        capsize=2.5, lw=1.4, label=gname)
        _zero(ax)
        ax.set_xticks(range(int(t.L.max()) + 1))
        ax.set_xlabel("lag of R&D growth (years)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("FE coefficient (95 % CI)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Timing of the R&D effect: negative on impact, positive "
                 "after 1-3 years", x=0.02, ha="left", fontsize=10)
    return _save(fig, outdir, "fig3_jcurve_lags")


def fig_event_study(results, outdir):
    """results: {(controls, detrended): EventStudyResult}."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True,
                             sharey=True)
    for ax, ((ctrl, dt), r) in zip(axes.flat, results.items()):
        t = pd.concat([r.by_event_time,
                       pd.DataFrame({"e": [-1], "att": [0.0], "lo": [0.0],
                                     "hi": [0.0]})]).sort_values("e")
        pre = t.e < 0
        for mask, col, mk in ((pre, GREY, "s"), (~pre, BLUE, "o")):
            s = t[mask]
            ax.errorbar(s.e, s.att, yerr=[s.att - s.lo, s.hi - s.att],
                        fmt=mk, color=col, capsize=2, ms=4.5, lw=1.2)
        _zero(ax)
        ax.axvline(-0.5, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{ctrl} controls{', detrended' if dt else ''}\n"
                     f"post ATT {r.overall_post:+.1f} %  ·  "
                     f"{'raw ' if dt else ''}pre-trend p = "
                     f"{r.pretrend_p:.3f}", loc="left", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("years relative to EU accession")
    for ax in axes[:, 0]:
        ax.set_ylabel("ATT, % of output per worker")
    fig.suptitle("EU accession event study (Callaway-Sant'Anna); grey = "
                 "pre-period, blue = post", x=0.02, ha="left", fontsize=10)
    return _save(fig, outdir, "fig4_event_study")


def fig_gmm(rows, outdir):
    """rows: DataFrame(measure, label, beta, se, valid)."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0))
    y = np.arange(len(rows))[::-1]
    for (meas, col, mk) in (("log_RD_stock_per_worker", BLUE, "o"),
                            ("RD_pct_GDP", ORANGE, "s")):
        m = (rows.measure == meas).to_numpy()
        for valid in (True, False):
            mm = m & (rows.valid.to_numpy() == valid)
            if not mm.any():
                continue
            ax.errorbar(rows.beta[mm], y[mm], xerr=1.96 * rows.se[mm],
                        fmt=mk, color=col, capsize=2.5, lw=1.2,
                        mfc=col if valid else "white", mec=col,
                        label=(f"{'log R&D stock/worker' if meas.startswith('log') else 'R&D % GDP'}"
                               f": {'valid' if valid else 'fails a test'}"))
    _zero(ax, "x")
    ax.set_yticks(y, rows.label, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("system-GMM coefficient on R&D (95 % CI)")
    ax.set_title("System GMM, R&D endogenous: filled = passes all "
                 "diagnostics", loc="left", fontsize=9.5)
    _legend_below(ax, ncol=2, y=-0.13)
    return _save(fig, outdir, "fig5_gmm_specifications")


def fig_dml(t, outdir):
    """t: DataFrame(Sample, Learner, slope, se, default)."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    t = t.assign(Sample=t.Sample.str.replace("Lux. + Malta dropped",
                                             "Lux.+Malta dropped"))
    samples = list(dict.fromkeys(t.Sample))
    for j, (lr, col, mk) in enumerate((("Random Forest", BLUE, "o"),
                                       ("Lasso", ORANGE, "s"))):
        s = t[t.Learner == lr]
        y = np.array([samples.index(x) for x in s.Sample]) + (j - 0.5) * 0.22
        ax.errorbar(s.slope, y, xerr=1.96 * s.se, fmt=mk, color=col,
                    capsize=2.5, lw=1.2, label=lr)
    _zero(ax, "x")
    ax.set_yticks(range(len(samples)), samples, fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"DML: $\partial\theta/\partial$ frontier gap (95 % CI)")
    ax.set_title("DML: R&D effect vs. frontier gap, by sample and learner",
                 loc="left", fontsize=9.5)
    _legend_below(ax, ncol=2, y=-0.2)
    return _save(fig, outdir, "fig6_dml_sensitivity")


def fig_beta(cs, fits, outdir):
    """cs: country table (Country, Group, initial, growth); fits by group.
    Left: all countries; right: the Innovative group zoomed, because its
    slope rests on a narrow range of initial levels."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.6),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, groups in ((a1, ("Emerging", "Innovative")), (a2, ("Innovative",))):
        for gname in groups:
            s = cs[cs.Group == gname]
            col, mk = GROUP_STYLE[gname]
            ax.scatter(s.initial, s.growth, c=col, marker=mk, s=34,
                       edgecolor="white", linewidth=0.8, zorder=3,
                       label=f"{gname}: β = {fits[gname]['beta']:+.2f} "
                             f"(p = {fits[gname]['p']:.3f})")
            xs = np.linspace(s.initial.min(), s.initial.max(), 20)
            ax.plot(xs, fits[gname]["a"] + fits[gname]["beta"] * xs,
                    color=col, lw=1.3)
        sub = cs[cs.Group.isin(groups)] if ax is a2 else \
            cs[cs.Group == "Emerging"]
        _label_points(ax, sub.initial.values, sub.growth.values,
                      list(sub.Country), fontsize=6)
        ax.set_xlabel("log output per worker, 2000")
    a1.set_ylabel("avg. annual growth 2000-2023 (%)")
    a1.set_title("All countries", loc="left")
    a2.set_title("Innovative (zoomed)", loc="left")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Absolute β-convergence within each typology", x=0.02,
                 ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig7_beta_convergence.{ext}"),
                    dpi=200, bbox_inches="tight", pad_inches=0.05,
                    metadata=_META[ext])
    plt.close(fig)
    return "fig7_beta_convergence"


def fig_tax_reforms(results, outdir):
    """results: {outcome label: EventStudyResult} for R&D tax reforms."""
    fig, axes = plt.subplots(1, len(results), figsize=(WIDTH, 3.2),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, r) in zip(axes, results.items()):
        t = pd.concat([r.by_event_time,
                       pd.DataFrame({"e": [-1], "att": [0.0], "lo": [0.0],
                                     "hi": [0.0]})]).sort_values("e")
        pre = t.e < 0
        for mask, col, mk in ((pre, GREY, "s"), (~pre, BLUE, "o")):
            q = t[mask]
            ax.errorbar(q.e, q.att, yerr=[q.att - q.lo, q.hi - q.att],
                        fmt=mk, color=col, capsize=2, ms=4.5, lw=1.2)
        _zero(ax)
        ax.axvline(-0.5, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{lab}\npost ATT {r.overall_post:+.1f} %  ·  "
                     f"pre-trend p = {r.pretrend_p:.2f}", loc="left",
                     fontsize=8.5)
        ax.set_xlabel("years relative to reform")
    axes[0].set_ylabel("ATT, % (95 % CI)")
    fig.suptitle(f"R&D tax-incentive reforms ({len(next(iter(results.values())).treated)}"
                 " countries), not-yet-treated controls", x=0.02, ha="left",
                 fontsize=10)
    return _save(fig, outdir, "fig8_tax_reforms")

