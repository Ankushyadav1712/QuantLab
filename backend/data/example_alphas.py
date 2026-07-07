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
    {
        "id": "momentum_12_1",
        "name": "12-1 momentum (skip recent month)",
        "category": "Momentum",
        "expression": "rank(momentum_252 - momentum_20)",
        "description": (
            "The academic 12-1 momentum factor (Jegadeesh & Titman, 1993): rank on the "
            "12-month return but subtract the most recent month, which tends to mean-revert "
            "and contaminate a pure trend signal. Shows that momentum fields can be combined "
            "arithmetically to reshape the lookback window."
        ),
        "recommended_settings": {
            "neutralization": "sector",
            "decay": 5,
            "transaction_cost_bps": 5,
        },
        "teaches": ["12-1 momentum", "skip-a-month construction", "momentum field arithmetic"],
    },
    {
        "id": "short_term_reversal_zscore",
        "name": "5-day price z-score reversal",
        "category": "Reversal",
        "expression": "-ts_zscore(close, 5)",
        "description": (
            "Fade recent price extremes: ts_zscore(close, 5) measures how many standard "
            "deviations today's close sits above its own 5-day mean; the leading minus bets "
            "on mean reversion. The z-score self-normalizes per name, so the ranking isn't "
            "dominated by naturally high-vol stocks."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": [
            "short-term reversal",
            "ts_zscore self-normalization",
            "sign flip for reversion",
        ],
    },
    {
        "id": "sector_relative_value",
        "name": "Industry-relative value (cheapness within peers)",
        "category": "Sector-relative",
        "expression": "group_zscore(-pe_ratio, industry_group)",
        "description": (
            "Value is only fair between peers — a 15x P/E is cheap for software, dear for a "
            "utility. Z-scores the negated P/E WITHIN each industry group so a name scores "
            "high only when cheap versus its own peers, stripping the cross-industry level "
            "differences a naive value factor loads on. Fundamentals lagged ~1Q (PIT proxy)."
        ),
        "recommended_settings": {
            "neutralization": "sector",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["peer-relative value", "group_zscore operator", "industry_group granularity"],
    },
    {
        "id": "fcf_yield_value",
        "name": "Free-cash-flow yield (cash value)",
        "category": "Fundamentals",
        "expression": "rank(fcf_yield)",
        "description": (
            "Rank on free-cash-flow yield (FCF / price) and go long the high-yield names. FCF "
            "is harder to manipulate than earnings, making it a cleaner value signal than P/E. "
            "Because fcf_yield is already a yield (higher = cheaper) you rank it directly with "
            "NO negation — contrast pe_ratio, which must be negated. Fundamentals lagged ~1Q."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["cash-flow value", "yield ranks directly (no negation)", "1Q PIT lag"],
    },
    {
        "id": "roa_quality_spread",
        "name": "ROA quality minus leverage",
        "category": "Fundamentals",
        "expression": "zscore(roa) - zscore(debt_to_equity)",
        "description": (
            "Long profitable, low-leverage firms; short unprofitable, highly-levered ones. "
            "Return on assets captures operating quality while debt-to-equity penalizes "
            "balance-sheet risk; z-scoring both puts them on one scale before differencing. "
            "Shows how to net a 'good' factor against a 'risk' factor. Fundamentals lagged ~1Q."
        ),
        "recommended_settings": {
            "neutralization": "sector",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["quality factor (ROA)", "netting quality vs leverage", "z-score differencing"],
    },
    {
        "id": "low_vol_defensive",
        "name": "Low-volatility anomaly (defensive)",
        "category": "Multi-factor",
        "expression": "-rank(realized_vol_120)",
        "description": (
            "The low-volatility anomaly (Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014): "
            "long the calmest names, short the most volatile. Ranks 120-day realized vol and "
            "flips the sign so low-vol names get top weight — a violation of the naive "
            "risk-return tradeoff, driven by leverage constraints and lottery demand."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["low-volatility anomaly", "sign-flipping a risk field", "defensive tilt"],
    },
    {
        "id": "value_quality_lowvol_composite",
        "name": "Value + quality + low-vol composite",
        "category": "Multi-factor",
        "expression": "zscore(fcf_yield) + zscore(roa) - zscore(realized_vol_60)",
        "description": (
            "Equal-weight blend of cheapness (FCF yield), profitability (ROA), and "
            "defensiveness (short realized vol). Each leg is z-scored so no single factor "
            "dominates, and the 'bad' factor (vol) is subtracted so LOW vol scores high. "
            "Blending weakly-correlated premia lifts the composite's information ratio."
        ),
        "recommended_settings": {
            "neutralization": "sector",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["3-factor composite", "z-score before blending", "sign-flip bad factors"],
    },
    {
        "id": "amihud_illiquidity_premium",
        "name": "Amihud illiquidity premium",
        "category": "Microstructure",
        "expression": "-rank(ts_mean(amihud, 20))",
        "description": (
            "Long the least-liquid names, short the most-liquid. The Amihud (2002) ratio "
            "measures price impact per dollar traded; investors demand extra return to hold "
            "illiquid stocks. Smooths 20-day Amihud before ranking and flips the sign so "
            "high-Amihud (illiquid) names get the long weight."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 10,
        },
        "teaches": ["illiquidity premium", "Amihud ratio", "smooth before ranking"],
    },
    {
        "id": "volume_confirmed_momentum",
        "name": "Volume-confirmed momentum",
        "category": "Microstructure",
        "expression": "rank(momentum_20) * rank(ts_rank(dollar_volume, 20))",
        "description": (
            "Go long trends the market is actually participating in. Multiplying "
            "rank(momentum_20) by rank(ts_rank(dollar_volume, 20)) acts as a soft AND — "
            "biggest weight to names both trending up AND trading on rising volume, filtering "
            "out thin, low-conviction drifts. ts_rank places today's volume in its own range."
        ),
        "recommended_settings": {
            "neutralization": "sector",
            "decay": 0,
            "transaction_cost_bps": 10,
        },
        "teaches": [
            "signal confirmation (soft AND)",
            "ts_rank in own history",
            "volume-confirmed trend",
        ],
    },
    {
        "id": "vix_regime_lowvol",
        "name": "Regime switch: low-vol in fear, momentum in calm",
        "category": "Conditional",
        "expression": "if_else(greater(vix, 25), rank(-realized_vol_60), rank(momentum_60))",
        "description": (
            "A two-state macro switch: when VIX is above 25 (fear regime) the alpha ranks by "
            "LOW 60-day vol (flight to safety); when VIX is calm it ranks by 60-day momentum. "
            "Momentum crashes in high-VIX panics while low-vol shines exactly then, so "
            "switching by regime harvests each factor where it historically pays."
        ),
        "recommended_settings": {
            "neutralization": "market",
            "decay": 0,
            "transaction_cost_bps": 5,
        },
        "teaches": ["regime switching", "if_else two-state strategy", "factor rotation by VIX"],
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
