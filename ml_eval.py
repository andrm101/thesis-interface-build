"""
ml_eval.py — Leakage-safe evaluation of ML models on a country × year panel.

Random K-fold on panel rows puts the same country (and neighbouring years of
it) in both training and test folds, so a model can score well by memorising
country levels. The schemes here keep evaluation honest:

  • "Grouped by country"  — GroupKFold: every test fold holds out whole
                            countries → "how well does it predict an unseen
                            country?" (default)
  • "Forward-chaining"    — train on earlier years, test on the next block
                            of years → "how well does it forecast?"
  • "Random K-fold"       — the old, leaky scheme; kept only as a reference
                            to show how much it inflates R².

Scaling is done inside each fold (Pipeline), so test-fold statistics never
leak into the scaler either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMES = ("Grouped by country", "Forward-chaining (time)",
           "Random K-fold (leaky)")


def _time_splits(years: np.ndarray, n_splits: int):
    """Expanding-window splits over unique years (train < test years)."""
    uy = np.unique(years)
    n_splits = max(2, min(n_splits, len(uy) - 1))
    blocks = np.array_split(uy, n_splits + 1)
    for k in range(1, len(blocks)):
        train_years = np.concatenate(blocks[:k])
        test_years = blocks[k]
        tr = np.flatnonzero(np.isin(years, train_years))
        te = np.flatnonzero(np.isin(years, test_years))
        if len(tr) and len(te):
            yield tr, te


def cv_splits(scheme: str, groups, years, n_splits: int, n_rows: int):
    groups, years = np.asarray(groups), np.asarray(years)
    if scheme == SCHEMES[0]:
        k = max(2, min(n_splits, len(np.unique(groups))))
        return list(GroupKFold(n_splits=k).split(np.zeros(n_rows), groups=groups))
    if scheme == SCHEMES[1]:
        return list(_time_splits(years, n_splits))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=42)
                .split(np.zeros(n_rows)))


def holdout_split(scheme: str, groups, years, n_rows: int,
                  test_size: float = 0.2):
    """Single train/test split for the predicted-vs-actual plot."""
    groups, years = np.asarray(groups), np.asarray(years)
    idx = np.arange(n_rows)
    if scheme == SCHEMES[0]:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                        random_state=42)
                      .split(idx, groups=groups))
        return tr, te
    if scheme == SCHEMES[1]:
        uy = np.unique(years)
        cut = uy[int(np.floor(len(uy) * (1 - test_size)))]
        return idx[years < cut], idx[years >= cut]
    return train_test_split(idx, test_size=test_size, random_state=42)


def evaluate(model, X: pd.DataFrame, y: pd.Series, groups, years,
             scheme: str = SCHEMES[0], n_splits: int = 5) -> dict:
    """Cross-validate and hold out *model* under *scheme*.

    Returns a dict with CV/holdout metrics, holdout predictions, the leaky
    random-K-fold R² for reference, and a scaler + model refitted on all
    rows (for feature importances and the Scenarios tab).
    """
    pipe = make_pipeline(StandardScaler(), clone(model))
    n = len(X)

    def _cv(splits):
        s = cross_validate(pipe, X, y, cv=splits,
                           scoring=("r2", "neg_mean_squared_error"))
        rmse = np.sqrt(-s["test_neg_mean_squared_error"])
        return s["test_r2"], rmse

    r2, rmse = _cv(cv_splits(scheme, groups, years, n_splits, n))
    leaky_r2, _ = _cv(cv_splits(SCHEMES[2], groups, years, n_splits, n))

    tr, te = holdout_split(scheme, groups, years, n)
    hold = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
    y_te = y.iloc[te]
    y_pred = hold.predict(X.iloc[te])

    scaler = StandardScaler().fit(X)
    final = clone(model).fit(scaler.transform(X), y)
    return {
        "model": final, "scaler": scaler,
        "y_te": y_te, "y_pred": y_pred,
        "cv_r2_mean": r2.mean(), "cv_r2_std": r2.std(),
        "cv_rmse_mean": rmse.mean(), "cv_rmse_std": rmse.std(),
        "random_cv_r2": leaky_r2.mean(),
        "test_r2": r2_score(y_te, y_pred),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "n_folds": len(r2), "scheme": scheme,
    }


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                                   groups, years, scheme: str = SCHEMES[0],
                                   n_splits: int = 5, n_repeats: int = 10,
                                   seed: int = 0) -> pd.DataFrame:
    """Permutation importance measured on HELD-OUT folds of *scheme*.

    Impurity or coefficient importances describe the training fit; here
    each feature's importance is the drop in held-out R² when that feature
    is shuffled in the test fold (whole countries for the grouped scheme),
    averaged over folds and repeats. Negative values mean the feature
    hurts out-of-sample prediction.
    """
    from sklearn.inspection import permutation_importance

    pipe = make_pipeline(StandardScaler(), clone(model))
    per_fold = []
    for tr, te in cv_splits(scheme, groups, years, n_splits, len(X)):
        fit = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(fit, X.iloc[te], y.iloc[te], scoring="r2",
                                   n_repeats=n_repeats, random_state=seed)
        per_fold.append(r.importances_mean)
    A = np.vstack(per_fold)
    return (pd.DataFrame({"feature": X.columns, "mean": A.mean(axis=0),
                          "std": A.std(axis=0),
                          "share_positive": (A > 0).mean(axis=0)})
            .sort_values("mean", ascending=False).reset_index(drop=True))

