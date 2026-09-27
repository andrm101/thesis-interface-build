"""
outliers.py — Z-score outlier screening for country × year panels.

Two levels of screening, both pure functions (no Tk) so they can be tested
and reused from any tab:

  • Country level   — z-score of each country's *mean* of a variable
                      against the cross-country distribution. A country
                      whose mean is extreme on at least one screened
                      variable is flagged and (if enforced) dropped.
  • Observation     — z-score of each country-year value against *that
                      country's own* time series (within-country), which
                      catches data spikes without penalising countries
                      that are simply at a different level.

Methods:
  • "classic" — (x − mean) / sd                       (default |z| > 3)
  • "robust"  — 0.6745 · (x − median) / MAD           (Iglewicz & Hoaglin
                                                        modified z, |z| > 3.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Intensive, model-relevant variables screened by default. Scale variables
# (Population, Labor, Patents) and cumulative levels (A) are left out: they
# would flag big or old economies for being big, not for being anomalous.
DEFAULT_SCREEN_VARS = [
    "Y by L", "Savings Percentage", "Human Capital Proxy",
    "Labor in research", "PIB towards research", "Patents per capita",
]

VAR_SETS    = ("Model variables", "All numeric")
LEVELS      = ("Countries", "Observations", "Both")
METHODS     = ("Classic z", "Robust (MAD)")
OBS_ACTIONS = ("Set to NaN", "Winsorize", "Drop row")


def default_screen_vars(df: pd.DataFrame) -> list[str]:
    cols = [c for c in DEFAULT_SCREEN_VARS if c in df.columns]
    if cols:
        return cols
    return [c for c in df.select_dtypes("number").columns if c != "Year"]


def _z(x: pd.DataFrame, robust: bool) -> pd.DataFrame:
    """Column-wise z-scores (population sd). NaN where spread is zero."""
    if robust:
        med = x.median()
        mad = (x - med).abs().median()
        return 0.6745 * (x - med) / mad.replace(0, np.nan)
    return (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)


def country_zscores(df: pd.DataFrame, cols: list[str],
                    robust: bool = False) -> pd.DataFrame:
    """Country × variable table of z-scores of country means."""
    return _z(df.groupby("Country")[cols].mean(), robust)


def _within_centre_spread(df: pd.DataFrame, cols: list[str], robust: bool):
    """Row-aligned (centre, spread) of each country's own series."""
    g = df.groupby("Country")[cols]
    if robust:
        centre = g.transform("median")
        spread = ((df[cols] - centre).abs()
                  .groupby(df["Country"]).transform("median") / 0.6745)
    else:
        centre = g.transform("mean")
        spread = g.transform(lambda s: s.std(ddof=0))
    return centre, spread.replace(0, np.nan)


def observation_zscores(df: pd.DataFrame, cols: list[str],
                        robust: bool = False) -> pd.DataFrame:
    """Row-aligned within-country z-scores of each observation."""
    centre, spread = _within_centre_spread(df, cols, robust)
    return (df[cols] - centre) / spread


@dataclass
class ScreenResult:
    df: pd.DataFrame
    cols: list[str]
    threshold: float
    method: str
    level: str
    obs_action: str
    country_z: pd.DataFrame | None = None
    flagged_countries: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    flagged_obs: pd.DataFrame | None = None   # Country, Year, Variable, Value, z

    def summary(self) -> str:
        parts = []
        if self.flagged_countries:
            parts.append("excluded " + ", ".join(self.flagged_countries))
        if self.flagged_obs is not None and len(self.flagged_obs):
            parts.append(f"{len(self.flagged_obs)} obs "
                         f"{self.obs_action.lower()}")
        return "; ".join(parts) or "no outliers"


def screen(df: pd.DataFrame, cols: list[str] | None = None,
           threshold: float = 3.0, method: str = "Classic z",
           level: str = "Countries", obs_action: str = "Set to NaN",
           min_countries: int = 3) -> ScreenResult:
    """Screen *df* and return the cleaned frame plus what was flagged.

    Country screening is skipped when fewer than *min_countries* remain,
    since a z-score across two units is meaningless.
    """
    cols   = [c for c in (cols or default_screen_vars(df)) if c in df.columns]
    robust = method.startswith("Robust")
    out    = df.copy()
    res    = ScreenResult(out, cols, threshold, method, level, obs_action)
    if not cols or "Country" not in df.columns:
        return res

    # Observation spikes first, so a single bad cell cannot drag a whole
    # country's mean past the threshold and get the country dropped.
    if level in ("Observations", "Both") and len(out):
        zo   = observation_zscores(out, cols, robust)
        mask = zo.abs() > threshold
        recs = []
        for v in cols:
            idx = mask.index[mask[v].fillna(False)]
            for i in idx:
                recs.append((out.at[i, "Country"], out.at[i, "Year"]
                             if "Year" in out.columns else None,
                             v, out.at[i, v], zo.at[i, v]))
        res.flagged_obs = pd.DataFrame(
            recs, columns=["Country", "Year", "Variable", "Value", "z"])
        if recs:
            if obs_action == "Drop row":
                out = out[~mask.any(axis=1)]
            elif obs_action == "Winsorize":
                centre, spread = _within_centre_spread(out, cols, robust)
                out[cols] = out[cols].clip(centre - threshold * spread,
                                           centre + threshold * spread)
            else:
                out[cols] = out[cols].mask(mask)

    if level in ("Countries", "Both") and out["Country"].nunique() >= min_countries:
        z = country_zscores(out, cols, robust)
        res.country_z = z
        for country, row in z.iterrows():
            hits = [(v, float(row[v])) for v in cols
                    if pd.notna(row[v]) and abs(row[v]) > threshold]
            if hits:
                res.flagged_countries[country] = hits
        out = out[~out["Country"].isin(res.flagged_countries)]

    res.df = out.reset_index(drop=True)
    return res
