"""
tabs/tab_clustering.py — K-Means clustering, elbow plot, PCA biplot.

Implements the K-Means typology reported in the thesis:
  • K = 2: Innovative leaders vs Emerging adopters.
  • Default features: GERD-intensity (PIB towards research),
    Labor in research, Patents per capita, Human Capital Proxy.
  • Auto-labels clusters by mean R&D intensity — highest R&D ⇒ Innovative.
  • PCA biplot renders country labels with loading arrows, both coloured
    by the assigned cluster.

Mixin: ClusterTabMixin
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure

import theme
from helpers import make_text, write, clear_txt, embed_figure


class ClusterTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_cluster(self, parent):
        ctrl = ttk.LabelFrame(parent,
                              text="K-Means Clustering by R&D Profile",
                              padding=10)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Cluster variables (Ctrl+click):").pack(
            side=tk.LEFT)
        ttk.Label(row0, text="  K:").pack(side=tk.LEFT, padx=10)
        self.k_val = tk.IntVar(value=2)
        _spk = tk.Spinbox(row0, from_=2, to=8, width=4,
                          textvariable=self.k_val,
                          bg=theme.WBG, fg=theme.FG,
                          insertbackground=theme.FG)
        _spk.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_spk)

        cl_frm = ttk.Frame(ctrl); cl_frm.pack(fill=tk.X, pady=3)
        sb_cl  = ttk.Scrollbar(cl_frm); sb_cl.pack(side=tk.RIGHT, fill=tk.Y)
        self.cluster_lb = tk.Listbox(
            cl_frm, selectmode=tk.MULTIPLE, height=4,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            font=("Segoe UI", 9), yscrollcommand=sb_cl.set,
            exportselection=False,
        )
        self.cluster_lb.pack(fill=tk.BOTH, expand=True)
        sb_cl.config(command=self.cluster_lb.yview)

        btn_row = ttk.Frame(ctrl); btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Elbow Plot",
                   command=self._run_elbow).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Run Clustering", style="Accent.TButton",
                   command=self._run_cluster).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="PCA Biplot",
                   command=self._run_pca).pack(side=tk.LEFT, padx=4)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.cluster_txt        = make_text(left, height=30, copy_root=self.root)
        self.cluster_plot_frame = right

    def _refresh_cluster_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.cluster_lb.delete(0, tk.END)
        default = {"PIB towards research", "Labor in research",
                   "Patents per capita", "Human Capital Proxy"}
        for c in num_cols:
            self.cluster_lb.insert(tk.END, c)
            if c in default:
                self.cluster_lb.select_set(tk.END)

    def _get_cluster_matrix(self):
        if not self._check_data():
            return None, None, None
        sel = [self.cluster_lb.get(i) for i in self.cluster_lb.curselection()]
        if not sel:
            messagebox.showwarning("Config", "Select cluster variables.")
            return None, None, None
        cmeans = self.df.groupby("Country")[sel].mean().dropna()
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(cmeans)
        return cmeans, X_sc, sel

    # ── Elbow plot ────────────────────────────────────────────────────────────
    def _run_elbow(self):
        cmeans, X_sc, _ = self._get_cluster_matrix()
        if cmeans is None:
            return
        for w in self.cluster_plot_frame.winfo_children():
            w.destroy()
        max_k    = min(10, len(cmeans))
        ks       = range(2, max_k)
        inertias = [
            KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc).inertia_
            for k in ks
        ]
        fig = Figure(figsize=(7.5, 4.3), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        ax.plot(list(ks), inertias,
                marker="o", color=theme.BLUE, linewidth=2.2, markersize=9,
                markerfacecolor=theme.TEAL, markeredgecolor=theme.BLUE)
        # Annotate the thesis choice K=2
        if 2 in ks:
            idx = list(ks).index(2)
            ax.annotate("K = 2  (thesis choice)",
                        xy=(2, inertias[idx]),
                        xytext=(2.8, inertias[idx] * 1.1),
                        color=theme.TEAL, fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=theme.TEAL,
                                        lw=1.2))
        ax.set_xlabel("Number of clusters  K")
        ax.set_ylabel("Within-cluster sum of squares (inertia)")
        ax.set_title("Elbow Method · Choose K at the bend", pad=10)
        fig.tight_layout()
        embed_figure(fig, self.cluster_plot_frame)

    # ── Auto-label helper ─────────────────────────────────────────────────────
    def _auto_label_clusters(self, cmeans, sel, k):
        """Return {cluster_id: label} with Innovative/Emerging for K=2."""
        if k != 2:
            return {c: f"Cluster {c}"
                    for c in sorted(cmeans["Cluster"].unique())}
        rd_kw   = {"research", "gerd", "pib towards", "patents", "rnd", "r&d"}
        rd_cols = [c for c in sel
                   if any(k_ in c.lower() for k_ in rd_kw)] or sel
        ids     = sorted(cmeans["Cluster"].unique())
        means   = {c: cmeans[cmeans["Cluster"] == c][rd_cols].mean().mean()
                   for c in ids}
        inn_id  = max(means, key=means.get)
        return {c: ("Innovative" if c == inn_id else "Emerging") for c in ids}

    # ── K-Means run ───────────────────────────────────────────────────────────
    def _run_cluster(self):
        cmeans, X_sc, sel = self._get_cluster_matrix()
        if cmeans is None:
            return
        k      = self.k_val.get()
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        cmeans = cmeans.copy(); cmeans["Cluster"] = labels + 1
        self.clusters = cmeans.reset_index()[["Country", "Cluster"]]

        cl_labels = self._auto_label_clusters(cmeans, sel, k)
        for w in self.cluster_plot_frame.winfo_children():
            w.destroy()
        clear_txt(self.cluster_txt)
        write(self.cluster_txt,
              f"K-MEANS CLUSTERING  (K={k})\n{'='*52}\n"
              f"Variables: {', '.join(sel)}\n"
              f"Inertia  : {km.inertia_:.2f}\n")
        if k == 2:
            write(self.cluster_txt,
                  "Labels   : Innovative (high R&D) / Emerging (low R&D)\n")

        for c in sorted(cmeans["Cluster"].unique()):
            lbl     = cl_labels[c]
            members = cmeans[cmeans["Cluster"] == c].index.tolist()
            write(self.cluster_txt,
                  f"\n{lbl}  ({len(members)} countries):\n"
                  f"  {', '.join(members)}\n  Means:\n")
            subset = cmeans[cmeans["Cluster"] == c][sel]
            for col in sel:
                write(self.cluster_txt,
                      f"    {col:<30} {subset[col].mean():>9.3f}\n")

        # Cluster profile chart — one subplot per cluster for readability
        fig     = Figure(figsize=(10, 5.2), facecolor=theme.BG)
        profile = cmeans.groupby("Cluster")[sel].mean()
        sc      = StandardScaler()
        prof_sc = pd.DataFrame(sc.fit_transform(profile),
                               index=profile.index, columns=profile.columns)
        pal = theme.cluster_palette()
        ax  = fig.add_subplot(111)
        x   = np.arange(len(sel))
        bw  = 0.8 / max(k, 1)
        for i, cl in enumerate(prof_sc.index):
            lbl = cl_labels[cl]
            ax.bar(x + i * bw, prof_sc.loc[cl], width=bw * 0.92,
                   label=lbl, color=pal[i % len(pal)], alpha=0.9,
                   edgecolor="none")
        ax.axhline(0, color=theme.GRAY, linewidth=0.8, linestyle="--")
        ax.set_xticks(x + bw * (k - 1) / 2)
        ax.set_xticklabels(sel, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Standardised cluster mean  (z-score)")
        ax.set_title(f"Cluster Profiles · K = {k}  ·  "
                     f"Innovative vs Emerging typology",
                     pad=10)
        ax.legend(loc="upper right", title="Cluster")
        fig.tight_layout()
        embed_figure(fig, self.cluster_plot_frame)
        summary = " | ".join(f"{v}: {k_}" for k_, v in cl_labels.items())
        self.status_var.set(
            f"Clustering done — K={k}, inertia={km.inertia_:.1f}  [{summary}]")

    # ── PCA biplot ────────────────────────────────────────────────────────────
    def _run_pca(self):
        cmeans, X_sc, sel = self._get_cluster_matrix()
        if cmeans is None:
            return
        k      = self.k_val.get()
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc) + 1
        pca    = PCA(n_components=2)
        X_pca  = pca.fit_transform(X_sc)
        ve     = pca.explained_variance_ratio_

        cl_labels = self._auto_label_clusters(
            pd.DataFrame({"Cluster": labels}, index=cmeans.index), sel, k)
        for w in self.cluster_plot_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(9.5, 6.5), facecolor=theme.BG)
        ax  = fig.add_subplot(111)
        pal = theme.cluster_palette()
        # Light decision grid behind the points
        ax.axhline(0, color=theme.GRAY, linewidth=0.6, linestyle="--",
                   alpha=0.7)
        ax.axvline(0, color=theme.GRAY, linewidth=0.6, linestyle="--",
                   alpha=0.7)
        for c in sorted(set(labels)):
            mask = labels == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       label=cl_labels[c],
                       color=pal[(c - 1) % len(pal)],
                       s=110, alpha=0.9, zorder=3,
                       edgecolor=theme.WBG, linewidth=1.2)
        for i, nm in enumerate(cmeans.index):
            ax.annotate(nm, (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=8, color=theme.FG, alpha=0.9,
                        xytext=(5, 5), textcoords="offset points",
                        fontweight="medium")
        # Loadings arrows
        load = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, var in enumerate(sel):
            ax.annotate(
                "", xy=(load[j, 0] * 0.55, load[j, 1] * 0.55),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=theme.AMBER,
                                lw=1.3, alpha=0.85),
            )
            ax.text(load[j, 0] * 0.62, load[j, 1] * 0.62,
                    var, color=theme.AMBER, fontsize=8,
                    fontweight="bold",
                    ha="left", va="bottom")
        ax.set_xlabel(f"Principal Component 1  ({ve[0]*100:.1f}% var.)")
        ax.set_ylabel(f"Principal Component 2  ({ve[1]*100:.1f}% var.)")
        ax.set_title(
            "PCA Biplot · EU-25 country typology by R&D profile",
            pad=10)
        ax.legend(loc="best", title="Cluster")
        fig.tight_layout()
        embed_figure(fig, self.cluster_plot_frame)
