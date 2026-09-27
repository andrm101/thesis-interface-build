"""
tabs/tab_ml.py  —  Machine Learning model training, evaluation, and diagnostics.
Models: Random Forest, Gradient Boosting, Ridge, Lasso.
Mixin: MLTabMixin
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from matplotlib.figure import Figure

import ml_eval
import theme
from helpers import make_text, write, clear_txt, embed_figure


class MLTabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — ML MODELS
    # ══════════════════════════════════════════════════════════════════════════
    def _tab_ml(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Machine Learning Models", padding=10)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        row0 = ttk.Frame(ctrl); row0.pack(fill=tk.X, pady=3)
        ttk.Label(row0, text="Target:").pack(side=tk.LEFT)
        self.ml_target = ttk.Combobox(row0, width=22, state="readonly")
        self.ml_target.pack(side=tk.LEFT, padx=6)
        ttk.Label(row0, text="  Model:").pack(side=tk.LEFT, padx=6)
        self.ml_model_type = ttk.Combobox(row0, width=26, state="readonly",
                                          values=[
            "Random Forest",
            "Gradient Boosting",
            "Ridge Regression",
            "Lasso Regression",
            "All Models (Compare)",
        ])
        self.ml_model_type.current(4)
        self.ml_model_type.pack(side=tk.LEFT, padx=4)

        row1 = ttk.Frame(ctrl); row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Features (Ctrl+click):").pack(side=tk.LEFT)
        ttk.Label(row1, text="  CV Folds:").pack(side=tk.LEFT, padx=10)
        self.cv_folds = tk.IntVar(value=5)
        _spcv = tk.Spinbox(row1, from_=3, to=10, width=4,
                           textvariable=self.cv_folds,
                           bg=theme.WBG, fg=theme.FG,
                           insertbackground=theme.FG)
        _spcv.pack(side=tk.LEFT, padx=4)
        self._tk_spinboxes.append(_spcv)
        ttk.Label(row1, text="  CV scheme:").pack(side=tk.LEFT, padx=(10, 0))
        self.cv_scheme = tk.StringVar(value=ml_eval.SCHEMES[0])
        ttk.Combobox(row1, textvariable=self.cv_scheme,
                     values=ml_eval.SCHEMES, state="readonly",
                     width=24).pack(side=tk.LEFT, padx=4)
        self.use_lags = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Include lag features (t−1)",
                        variable=self.use_lags).pack(side=tk.LEFT, padx=10)

        feat_frm = ttk.Frame(ctrl); feat_frm.pack(fill=tk.X, pady=3)
        sb_f = ttk.Scrollbar(feat_frm); sb_f.pack(side=tk.RIGHT, fill=tk.Y)
        self.feat_lb = tk.Listbox(
            feat_frm, selectmode=tk.MULTIPLE, height=4,
            bg=theme.WBG, fg=theme.FG, selectbackground=theme.BLUE,
            font=("Segoe UI", 9), yscrollcommand=sb_f.set,
            exportselection=False,
        )
        self.feat_lb.pack(fill=tk.BOTH, expand=True)
        sb_f.config(command=self.feat_lb.yview)

        btn_row = ttk.Frame(ctrl); btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Refresh",
                   command=self._refresh_ml_vars).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Train & Evaluate",
                   style="Accent.TButton",
                   command=self._run_ml).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Feature Importance",
                   command=self._plot_importance).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Actual vs Predicted",
                   command=self._plot_avp).pack(side=tk.LEFT, padx=4)

        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        left  = ttk.Frame(pane); pane.add(left,  weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=2)
        self.ml_txt        = make_text(left, height=30, copy_root=self.root)
        self.ml_plot_frame = right

    # ── Variable refresh ──────────────────────────────────────────────────────
    def _refresh_ml_vars(self):
        if self.df is None:
            return
        num_cols = [c for c in self.df.select_dtypes("number").columns
                    if c != "Year"]
        self.ml_target["values"] = num_cols
        self.ml_target.set("Y by L" if "Y by L" in num_cols
                           else (num_cols[0] if num_cols else ""))
        default = {"Savings Percentage", "Human Capital Proxy",
                   "Labor in research", "PIB towards research",
                   "Patents per capita", "Labor not in research",
                   "A", "TFP Growth Rate"}
        self.feat_lb.delete(0, tk.END)
        target = self.ml_target.get()
        for c in num_cols:
            if c == target:
                continue
            self.feat_lb.insert(tk.END, c)
            if c in default:
                self.feat_lb.select_set(tk.END)

    # ── Data prep ─────────────────────────────────────────────────────────────
    def _prepare_ml_data(self):
        target = self.ml_target.get()
        feats  = [self.feat_lb.get(i) for i in self.feat_lb.curselection()]
        if not target or not feats:
            messagebox.showwarning("Config", "Select target and features.")
            return None, None, None, None
        df = self.df[["Country", "Year", target] + feats].copy()
        if self.use_lags.get():
            df = df.sort_values(["Country", "Year"])
            for f in feats:
                df[f"{f}_lag1"] = df.groupby("Country")[f].shift(1)
        df = df.dropna().reset_index(drop=True)
        meta = df[["Country", "Year"]]
        X  = df.drop(columns=["Country", "Year", target])
        y  = df[target]
        return X, y, list(X.columns), meta

    # ── Training ──────────────────────────────────────────────────────────────
    def _run_ml(self):
        if not self._check_data():
            return
        X, y, feat_names, meta = self._prepare_ml_data()
        if X is None:
            return
        scheme = self.cv_scheme.get()
        folds  = int(self.cv_folds.get())
        self.status_var.set("Training ML models…")

        def run():
            try:
                mtype = self.ml_model_type.get()
                pool  = {}
                if mtype in ("Random Forest",     "All Models (Compare)"):
                    pool["Random Forest"]     = RandomForestRegressor(
                        n_estimators=300, max_depth=8,
                        random_state=42, n_jobs=-1)
                if mtype in ("Gradient Boosting", "All Models (Compare)"):
                    pool["Gradient Boosting"] = GradientBoostingRegressor(
                        n_estimators=300, max_depth=4,
                        learning_rate=0.05, random_state=42)
                if mtype in ("Ridge Regression",  "All Models (Compare)"):
                    pool["Ridge"]             = Ridge(alpha=1.0)
                if mtype in ("Lasso Regression",  "All Models (Compare)"):
                    pool["Lasso"]             = Lasso(alpha=0.01,
                                                      max_iter=5000)
                if not pool:
                    pool["Random Forest"] = RandomForestRegressor(
                        n_estimators=200, random_state=42)

                results = {}
                for name, model in pool.items():
                    r = ml_eval.evaluate(model, X, y, meta["Country"],
                                         meta["Year"], scheme, folds)
                    r["feat_names"] = feat_names
                    results[name] = r

                self.ml_results  = results
                self.ml_features = feat_names
                best             = max(results,
                                       key=lambda k: results[k]["cv_r2_mean"])
                self.ml_model    = results[best]["model"]
                self.ml_scaler   = results[best]["scaler"]
                self.root.after(0, lambda: self._display_ml(results))
            except Exception:
                import traceback as tb
                msg = tb.format_exc()
                self.root.after(
                    0, lambda: messagebox.showerror("ML Error", msg))

        threading.Thread(target=run, daemon=True).start()

    def _display_ml(self, results):
        clear_txt(self.ml_txt)
        best  = max(results, key=lambda k: results[k]["cv_r2_mean"])
        any_r = next(iter(results.values()))
        W = 76
        lines = [
            f"ML MODEL EVALUATION — {any_r['scheme']}\n{'='*W}\n",
            f"{'Model':<20} {'CV R²':>8} {'±':>6} {'Hold R²':>9} "
            f"{'RMSE':>9} {'Random-CV R²':>14}\n",
            f"{'─'*W}\n",
        ]
        for name, r in results.items():
            lines.append(
                f"{name:<20} {r['cv_r2_mean']:>8.4f} {r['cv_r2_std']:>6.4f} "
                f"{r['test_r2']:>9.4f} {r['test_rmse']:>9.4f} "
                f"{r['random_cv_r2']:>14.4f}"
                + ("  ← best\n" if name == best else "\n"))
        lines.append(f"{'─'*W}\n")
        lines.append(
            f"CV folds: {any_r['n_folds']}  |  Holdout: 20 % "
            f"({'countries' if any_r['scheme'] == ml_eval.SCHEMES[0] else 'final years' if any_r['scheme'] == ml_eval.SCHEMES[1] else 'rows'})"
            f"  |  Lag features: {self.use_lags.get()}\n"
            "Best model chosen by CV R². Scaling is fitted inside each fold.\n"
            "Random-CV R² (shuffled rows, same country in train & test) is "
            "shown only\nfor reference: the gap to CV R² is the leakage it "
            "hides.\n")
        write(self.ml_txt, "".join(lines))
        self.status_var.set(
            f"ML complete — best: {best}  CV R²={results[best]['cv_r2_mean']:.4f}")

        if len(results) > 1:
            for w in self.ml_plot_frame.winfo_children():
                w.destroy()
            names  = list(results.keys())
            colors = [theme.TEAL if n == best else theme.BLUE for n in names]
            fig    = Figure(figsize=(10, 5), facecolor=theme.BG)
            ax1    = fig.add_subplot(121)
            xs = np.arange(len(names))
            ax1.bar(xs - 0.2, [results[n]["cv_r2_mean"] for n in names],
                    width=0.4, color=colors, alpha=0.9,
                    yerr=[results[n]["cv_r2_std"] for n in names],
                    label="Honest CV")
            ax1.bar(xs + 0.2, [results[n]["random_cv_r2"] for n in names],
                    width=0.4, color=theme.GRAY, alpha=0.6,
                    label="Random K-fold (leaky)")
            ax1.set_xticks(xs, names)
            ax1.axhline(0, color=theme.FG, lw=0.6)
            ax1.legend(fontsize=7)
            ax1.set_title("CV R²", color=theme.FG)
            ax1.tick_params(axis="x", rotation=20, colors=theme.FG, labelsize=8)
            ax2    = fig.add_subplot(122)
            ax2.bar(names, [results[n]["test_rmse"] for n in names],
                    color=colors, alpha=0.85)
            ax2.set_title("Holdout RMSE", color=theme.FG)
            ax2.tick_params(axis="x", rotation=20, colors=theme.FG, labelsize=8)
            fig.suptitle("Model Comparison", color=theme.FG)
            fig.tight_layout()
            embed_figure(fig, self.ml_plot_frame)

    # ── Feature importance ────────────────────────────────────────────────────
    def _plot_importance(self):
        if not self.ml_results:
            messagebox.showinfo("Info", "Train models first.")
            return
        for w in self.ml_plot_frame.winfo_children():
            w.destroy()
        tree_m   = {n: r for n, r in self.ml_results.items()
                    if hasattr(r["model"], "feature_importances_")}
        linear_m = {n: r for n, r in self.ml_results.items()
                    if hasattr(r["model"], "coef_")}
        all_m    = {**tree_m, **linear_m}
        if not all_m:
            messagebox.showinfo("Info", "No models with importances.")
            return

        n   = len(all_m)
        fig = Figure(figsize=(10, max(4, 4 * n)), facecolor=theme.BG)
        for idx, (name, r) in enumerate(all_m.items(), 1):
            ax    = fig.add_subplot(n, 1, idx)
            feats = r["feat_names"]
            if name in tree_m:
                imp   = r["model"].feature_importances_
                mask  = imp >= 0.02
                ff    = [feats[i] for i in range(len(feats)) if mask[i]]
                fi    = imp[mask]
                order = np.argsort(fi)
                ax.barh([ff[j] for j in order], fi[order],
                        color=theme.TEAL, alpha=0.85)
                ax.set_title(f"{name} — Feature Importance (≥2%)",
                             color=theme.FG, fontsize=9)
            else:
                coef  = r["model"].coef_
                order = np.argsort(np.abs(coef))
                ax.barh([feats[j] for j in order], coef[order],
                        color=theme.BLUE, alpha=0.85)
                ax.axvline(0, color=theme.RED, linewidth=0.8, linestyle="--")
                ax.set_title(f"{name} — Coefficients",
                             color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=8)
        fig.tight_layout()
        embed_figure(fig, self.ml_plot_frame)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    def _plot_avp(self):
        if not self.ml_results:
            messagebox.showinfo("Info", "Train models first.")
            return
        for w in self.ml_plot_frame.winfo_children():
            w.destroy()
        n    = len(self.ml_results)
        cols = min(2, n); rows = (n + 1) // 2
        fig  = Figure(figsize=(10, 4 * rows), facecolor=theme.BG)
        for i, (name, r) in enumerate(self.ml_results.items(), 1):
            ax  = fig.add_subplot(rows, cols, i)
            yte = r["y_te"]; yp = r["y_pred"]
            ax.scatter(yte, yp, color=theme.BLUE, alpha=0.6, s=30, zorder=3)
            mn = min(yte.min(), yp.min()); mx = max(yte.max(), yp.max())
            ax.plot([mn, mx], [mn, mx],
                    color=theme.TEAL, linewidth=1.5, linestyle="--")
            ax.set_xlabel("Actual",    color=theme.FG, fontsize=8)
            ax.set_ylabel("Predicted", color=theme.FG, fontsize=8)
            ax.set_title(
                f"{name}\nR²={r['test_r2']:.4f}  RMSE={r['test_rmse']:.4f}",
                color=theme.FG, fontsize=9)
            ax.tick_params(colors=theme.FG, labelsize=7)
        fig.suptitle("Actual vs Predicted", color=theme.FG)
        fig.tight_layout()
        embed_figure(fig, self.ml_plot_frame)
