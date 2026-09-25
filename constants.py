"""
constants.py — Application-wide lookup tables and significance helpers.

All country groupings here reflect the thesis sample (EU-25, 1998–2023) and
the two-cluster K-Means typology reported in the dissertation:

    • Innovation Leaders ("Innovative") — Nordic and Central European
      economies with high R&D intensity and strong absorptive capacity.
    • Emerging Adopters  ("Emerging")   — Eastern and Southern European
      economies with lower R&D intensity and smaller innovation footprints.
"""

# ── Broad legacy groupings (kept for filter shortcuts) ─────────────────────
WESTERN_EU = [
    'austria', 'belgium', 'denmark', 'finland', 'france', 'germany', 'greece',
    'iceland', 'ireland', 'italy', 'luxembourg', 'netherlands', 'norway',
    'portugal', 'spain', 'sweden', 'switzerland', 'united kingdom', 'uk',
    'malta', 'cyprus',
]

EASTERN_EU = [
    'albania', 'bosnia and herzegovina', 'bulgaria', 'croatia', 'czech republic',
    'czechia', 'estonia', 'hungary', 'latvia', 'lithuania', 'montenegro',
    'north macedonia', 'poland', 'romania', 'serbia', 'slovakia',
    'slovak republic', 'slovenia', 'belarus', 'moldova', 'russia',
    'russian federation', 'ukraine',
]

# ── Thesis-specific country clusters (EU-25 sample) ────────────────────────
# These are the a-priori labels used to validate the K-Means output. They
# mirror the typology reported in the dissertation's findings.
INNOVATIVE_CLUSTER = [
    # Nordic
    'denmark', 'finland', 'sweden',
    # Central / DACH / Benelux
    'austria', 'belgium', 'france', 'germany', 'luxembourg',
    'netherlands', 'ireland',
]

EMERGING_CLUSTER = [
    # Eastern EU
    'bulgaria', 'croatia', 'czechia', 'czech republic', 'estonia', 'hungary',
    'latvia', 'lithuania', 'poland', 'romania', 'slovakia', 'slovak republic',
    'slovenia',
    # Southern EU
    'greece', 'italy', 'portugal', 'spain',
    # Small Mediterranean
    'cyprus', 'malta',
]

EU25 = sorted(set(INNOVATIVE_CLUSTER + EMERGING_CLUSTER))


# ── Variable role detection (panel/econometric helpers) ────────────────────
# Keywords used across tabs to auto-detect variables that play a specific
# economic role.  Lower-case substring match against column names.
VAR_ROLES = {
    "output_per_worker":  ("y by l", "y/l", "productivity", "output per worker"),
    "savings":            ("savings", "saving rate", "gross savings"),
    "human_capital":      ("human capital", "schooling", "education", "hc"),
    "rnd_intensity":      ("pib towards research", "gerd", "r&d", "rnd",
                           "research intensity", "labor in research"),
    "patents":            ("patents", "patent applications"),
    "initial_gdp":        ("initial gdp", "gdp_0", "gdp_initial", "log_gdp0"),
    "tfp":                ("tfp", "total factor productivity", "a"),
}


def detect_variable_role(col: str) -> str | None:
    """Return the role key for a column name, or None if no match."""
    name = col.lower()
    for role, keywords in VAR_ROLES.items():
        if any(kw in name for kw in keywords):
            return role
    return None


# ── Significance ───────────────────────────────────────────────────────────
SIG_LEVELS = [(0.01, "***"), (0.05, "**"), (0.1, "*")]


def stars(p: float) -> str:
    """Return APA-style significance stars for a p-value."""
    for threshold, symbol in SIG_LEVELS:
        if p < threshold:
            return symbol
    return ""
