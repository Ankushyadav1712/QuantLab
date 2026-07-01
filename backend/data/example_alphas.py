"""Curated example alphas for the Load Example dropdown.

Each entry pairs a complete expression with its recommended settings and a
short description of what it teaches.  Surfaced via /api/examples.

Adding a new example:
    1. Append a dict to EXAMPLE_ALPHAS with all six fields.
    2. Make sure the expression parses + lints clean against the current
       operator/field set (test_examples.py guards this).
    3. Pick a category that already exists if possible.
"""

from __future__ import annotations

from typing import Any

EXAMPLE_ALPHAS: list[dict[str, Any]] = [
    {
        "id": "momentum_20_decayed",
        "name": "Momentum 20-day (decay-smoothed)",
        "category": "Momentum",
        "expression": "decay_linear(rank(momentum_20), 20)",
        "description": (
            "Long-only 20-day momentum signal smoothed with a left-weighted "
            "moving average to cut turnover. Sharpe ~1.0 on the default "
            "S&P 100 universe over 2019–2024."
        ),
        "recommended_settings": {
            "neutralization": "none",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["momentum factor", "decay_linear smoothing", "turnover reduction"],
    },
    {
        "id": "reversal_5_volscaled",
        "name": "5-day reversal, vol-scaled",
        "category": "Reversal",
        "expression": "rank(reversal_5) / (realized_vol + 0.001)",
        "description": (
            "Short-horizon mean-reversion signal scaled inversely by realized "
            "volatility — bet against names that ran up most when their vol "
            "is low (cleaner reversion than vol-driven noise)."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["reversal effect", "vol scaling", "market neutralization"],
    },
    {
        "id": "close_to_high",
        "name": "Distance from 52-week high",
        "category": "Momentum",
        "expression": "rank(close_to_high_252)",
        "description": (
            "Names trading near their 52-week high have outperformed historically "
            "(George & Hwang, 2004). A clean way to capture trend strength."
        ),
        "recommended_settings": {
            "neutralization": "none",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["52-week high anomaly", "rank operator", "trend signals"],
    },
    {
        "id": "sector_neutral_momentum",
        "name": "Sector-neutral 60-day momentum",
        "category": "Sector-relative",
        "expression": "group_neutralize(rank(momentum_60), sector)",
        "description": (
            "Removes sector tilt — you're betting on the best name in each "
            "sector, not the best sector. Cleaner exposure to stock-specific "
            "alpha at the cost of giving up sector-rotation signal."
        ),
        "recommended_settings": {
            "neutralization": "none",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["group operators", "GICS data fields", "neutralization"],
    },
    {
        "id": "pe_meanreversion",
        "name": "P/E mean reversion (cheapness)",
        "category": "Fundamentals",
        "expression": "-zscore(pe_ratio)",
        "description": (
            "Low P/E → long, high P/E → short. The classic value factor. Note "
            "fundamentals are lagged 1 quarter as a PIT proxy, so this won't "
            "react to fresh earnings until the next quarter."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["value factor", "fundamentals fields", "1Q PIT lag"],
    },
    {
        "id": "quality_momentum_combo",
        "name": "Quality + momentum combo",
        "category": "Multi-factor",
        "expression": "zscore(roe) + zscore(momentum_60)",
        "description": (
            "Equal-weight blend of quality (ROE) and price momentum. Each "
            "z-scored so the two factors contribute on the same scale. "
            "Classic multi-factor stock-picking pattern."
        ),
        "recommended_settings": {
            "neutralization": "sector",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["multi-factor blending", "z-score normalization", "sector neutralization"],
    },
    {
        "id": "vix_conditional_momentum",
        "name": "Momentum, only when VIX is calm",
        "category": "Conditional",
        "expression": "trade_when(less(vix, 20), rank(momentum_20))",
        "description": (
            "Take momentum positions only when implied vol is below 20. "
            "Carries the previous position forward when VIX spikes — cuts "
            "drawdowns at the cost of missing some momentum continuations."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["macro conditioning", "trade_when operator", "VIX regimes"],
    },
    {
        "id": "yield_curve_reversal",
        "name": "Reversal during curve inversion",
        "category": "Conditional",
        "expression": "when(less(term_spread_10y_2y, 0), rank(reversal_5))",
        "description": (
            "Run a short-horizon reversal alpha only when the 10y-2y Treasury "
            "spread is inverted (recession signal). Outside the regime the "
            "alpha is NaN (no position). A regime-conditional bet."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["regime conditioning", "macro fields", "when operator"],
    },
    {
        "id": "volume_orthogonal_close",
        "name": "Volume-orthogonal close rank",
        "category": "Microstructure",
        "expression": "vector_neut(rank(close), rank(volume))",
        "description": (
            "Rank stocks by close price, then orthogonalize against volume "
            "rank. The residual captures the part of close-rank that isn't "
            "explained by volume — useful when you want to remove a known "
            "factor exposure."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["vector_neut orthogonalization", "factor neutralization"],
    },
    {
        "id": "concentrated_top10",
        "name": "Top-10 concentrated momentum",
        "category": "Conditional",
        "expression": "keep(rank(momentum_60), 10)",
        "description": (
            "Take only the 10 strongest momentum signals; zero out the rest. "
            "Concentrates the alpha into its highest-conviction names. "
            "Higher per-name concentration risk in exchange for a sharper bet."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["keep operator", "concentration vs. diversification"],
    },
]


def list_examples() -> list[dict[str, Any]]:
    """Returned by /api/examples — a copy so the route handler can't mutate the source."""
    return [dict(e) for e in EXAMPLE_ALPHAS]


def get_example(example_id: str) -> dict[str, Any] | None:
    for e in EXAMPLE_ALPHAS:
        if e["id"] == example_id:
            return dict(e)
    return None
