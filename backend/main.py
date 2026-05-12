from __future__ import annotations

import hmac
import json
import logging
import math
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from analytics.factor_decomp import FactorDecomposition
from analytics.performance import PerformanceAnalytics, _safe_float, _safe_list
from config import (
    ALLOWED_ORIGINS,
    DATA_END,
    DATA_START,
    DEFAULT_BOOKSIZE,
    ENVIRONMENT,
    SECTOR_MAP,
)
from data.example_alphas import get_example, list_examples
from data.factors import download_ff5_daily
from data.fetcher import ALL_FIELDS, DataFetcher
from data.fundamentals import (
    LAG_QUARTERS,
    download_fundamentals,
)
from data.macro import (
    broadcast_to_matrix as macro_broadcast,
)
from data.macro import (
    download_macro,
)
from data.sp100_history import membership_summary
from data.universes import (
    all_tickers as universe_all_tickers,
)
from data.universes import (
    available_neutralizations,
    default_universe_id,
    get_universe,
    gics_data_frames,
    gics_for,
    list_universes,
)
from db.database import connect
from db.migrations import init_db
from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator
from engine.lint import lint_ast
from engine.parser import Parser
from engine.sweep import combo_for_index, expand_sweeps
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.schemas import (
    AlphaSaveRequest,
    CompareRequest,
    CorrelationRequest,
    MultiAlphaRequest,
    SimulationRequest,
    SweepRequest,
    ValidateRequest,
)
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

OPERATORS = [
    # ----- Time-series (per ticker, axis=0) -----
    {
        "name": "ts_mean",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling mean of x over d periods",
    },
    {
        "name": "ts_std",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling stdev of x over d periods",
    },
    {
        "name": "ts_min",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling min over d periods",
    },
    {
        "name": "ts_max",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling max over d periods",
    },
    {
        "name": "ts_sum",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling sum over d periods",
    },
    {
        "name": "ts_rank",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling percentile rank in [0,1] over d periods",
    },
    {
        "name": "ts_median",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling median",
    },
    {
        "name": "ts_skewness",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling skewness",
    },
    {
        "name": "ts_kurtosis",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling excess kurtosis",
    },
    {
        "name": "ts_zscore",
        "args": "(x, d)",
        "category": "time_series",
        "description": "(x - rolling_mean) / rolling_std",
    },
    {
        "name": "ts_quantile",
        "args": "(x, d, q=0.5)",
        "category": "time_series",
        "description": "Rolling q-th quantile",
    },
    {
        "name": "ts_arg_max",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Days-ago index of rolling max (0 = today)",
    },
    {
        "name": "ts_arg_min",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Days-ago index of rolling min (0 = today)",
    },
    {
        "name": "ts_product",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling product over d periods",
    },
    {
        "name": "ts_returns",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Multi-period simple return: x / x.shift(d) - 1",
    },
    {
        "name": "ts_decay_exp",
        "args": "(x, d, factor=0.5)",
        "category": "time_series",
        "description": "Exponentially-weighted MA over d periods",
    },
    {
        "name": "ts_partial_corr",
        "args": "(x, y, z, d)",
        "category": "time_series",
        "description": "Rolling partial correlation of x,y controlling for z",
    },
    {
        "name": "ts_regression",
        "args": "(y, x, d)",
        "category": "time_series",
        "description": "Rolling OLS slope: cov(x,y) / var(x)",
    },
    {
        "name": "ts_min_max_diff",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Rolling (max - min); volatility proxy",
    },
    {"name": "delta", "args": "(x, d)", "category": "time_series", "description": "x - x.shift(d)"},
    {"name": "delay", "args": "(x, d)", "category": "time_series", "description": "x.shift(d)"},
    {
        "name": "decay_linear",
        "args": "(x, d)",
        "category": "time_series",
        "description": "Linearly weighted MA over d periods",
    },
    {
        "name": "ts_corr",
        "args": "(x, y, d)",
        "category": "time_series",
        "description": "Rolling Pearson correlation over d periods",
    },
    {
        "name": "ts_cov",
        "args": "(x, y, d)",
        "category": "time_series",
        "description": "Rolling covariance over d periods",
    },
    {
        "name": "days_from_last_change",
        "args": "(x)",
        "category": "time_series",
        "description": "Days since the value last changed (per ticker)",
    },
    {
        "name": "hump",
        "args": "(x, threshold=0.01)",
        "category": "time_series",
        "description": "Smoothing: only update when |x - prev| > threshold",
    },
    # ----- Cross-sectional (per date, axis=1) -----
    {
        "name": "rank",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Cross-sectional percentile rank in [0,1] per date",
    },
    {
        "name": "zscore",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Cross-sectional z-score per date",
    },
    {
        "name": "demean",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Subtract cross-sectional mean per date",
    },
    {
        "name": "scale",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Scale so |sum|=1 per date",
    },
    {
        "name": "normalize",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Demean then scale to unit |sum|",
    },
    {
        "name": "winsorize",
        "args": "(x, std=4)",
        "category": "cross_sectional",
        "description": "Clip each row at ±std·sigma around the row mean",
    },
    {
        "name": "quantile",
        "args": "(x, q=0.5)",
        "category": "cross_sectional",
        "description": "Per-row q-th quantile broadcast across columns",
    },
    {
        "name": "vector_neut",
        "args": "(x, y)",
        "category": "cross_sectional",
        "description": "Project x orthogonal to y (per row)",
    },
    {
        "name": "regression_neut",
        "args": "(x, y)",
        "category": "cross_sectional",
        "description": "Cross-sectional OLS residual of x on y",
    },
    {
        "name": "bucket",
        "args": "(x, n=5)",
        "category": "cross_sectional",
        "description": "Discretize each row into n equal-frequency buckets [0..n-1]",
    },
    {
        "name": "tail",
        "args": "(x, lower, upper, replace)",
        "category": "cross_sectional",
        "description": "Replace cells outside [lower, upper] with `replace`",
    },
    {
        "name": "kth_element",
        "args": "(x, k)",
        "category": "cross_sectional",
        "description": "Per-row k-th smallest (k=-1 for max)",
    },
    {
        "name": "harmonic_mean",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Per-row harmonic mean broadcast across columns",
    },
    {
        "name": "geometric_mean",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Per-row geometric mean broadcast across columns",
    },
    {
        "name": "step",
        "args": "(x)",
        "category": "cross_sectional",
        "description": "Per-row linear ramp from -1 to +1 by rank",
    },
    # ----- Group operators (require a group label like `sector`) -----
    {
        "name": "group_rank",
        "args": "(x, group)",
        "category": "group",
        "description": "Per row, percentile-rank x within each group",
    },
    {
        "name": "group_zscore",
        "args": "(x, group)",
        "category": "group",
        "description": "Per row, z-score within each group",
    },
    {
        "name": "group_neutralize",
        "args": "(x, group)",
        "category": "group",
        "description": "Subtract the group mean from each member",
    },
    {
        "name": "group_mean",
        "args": "(x, group)",
        "category": "group",
        "description": "Broadcast group mean to each member",
    },
    {
        "name": "group_sum",
        "args": "(x, group)",
        "category": "group",
        "description": "Broadcast group sum to each member",
    },
    {
        "name": "group_count",
        "args": "(x, group)",
        "category": "group",
        "description": "Broadcast group count to each member",
    },
    {
        "name": "group_max",
        "args": "(x, group)",
        "category": "group",
        "description": "Broadcast group max to each member",
    },
    {
        "name": "group_min",
        "args": "(x, group)",
        "category": "group",
        "description": "Broadcast group min to each member",
    },
    {
        "name": "group_normalize",
        "args": "(x, group)",
        "category": "group",
        "description": "Demean within group, scale to |sum|=1 per group",
    },
    {
        "name": "group_scale",
        "args": "(x, group)",
        "category": "group",
        "description": "Scale so |sum|=1 within each group",
    },
    # ----- Arithmetic / element-wise -----
    {
        "name": "abs",
        "args": "(x)",
        "category": "arithmetic",
        "description": "Element-wise absolute value",
    },
    {
        "name": "log",
        "args": "(x)",
        "category": "arithmetic",
        "description": "Element-wise natural log (NaN for x<=0)",
    },
    {"name": "exp", "args": "(x)", "category": "arithmetic", "description": "Element-wise e^x"},
    {
        "name": "sqrt",
        "args": "(x)",
        "category": "arithmetic",
        "description": "Element-wise sqrt (NaN for x<0)",
    },
    {
        "name": "mod",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "x mod y element-wise (NaN for y=0)",
    },
    {
        "name": "sign",
        "args": "(x)",
        "category": "arithmetic",
        "description": "Element-wise sign (-1, 0, +1)",
    },
    {
        "name": "power",
        "args": "(x, n)",
        "category": "arithmetic",
        "description": "x ** n element-wise",
    },
    {
        "name": "signed_power",
        "args": "(x, n)",
        "category": "arithmetic",
        "description": "sign(x) * |x|^n; preserves sign",
    },
    {
        "name": "sigmoid",
        "args": "(x)",
        "category": "arithmetic",
        "description": "Logistic sigmoid 1/(1+e^-x)",
    },
    {
        "name": "clip",
        "args": "(x, lo, hi)",
        "category": "arithmetic",
        "description": "Clamp each cell to [lo, hi]",
    },
    {"name": "max", "args": "(x, y)", "category": "arithmetic", "description": "Element-wise max"},
    {"name": "min", "args": "(x, y)", "category": "arithmetic", "description": "Element-wise min"},
    {
        "name": "replace",
        "args": "(x, old, new)",
        "category": "arithmetic",
        "description": "Replace cells equal to `old` with `new`",
    },
    {
        "name": "isnan",
        "args": "(x)",
        "category": "arithmetic",
        "description": "1.0 where cell is NaN, else 0.0",
    },
    {
        "name": "equal",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "Element-wise equality indicator (0/1)",
    },
    {
        "name": "less",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "Element-wise x < y; returns 1.0 / 0.0",
    },
    {
        "name": "greater",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "Element-wise x > y; returns 1.0 / 0.0",
    },
    {
        "name": "less_eq",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "Element-wise x <= y; returns 1.0 / 0.0",
    },
    {
        "name": "greater_eq",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "Element-wise x >= y; returns 1.0 / 0.0",
    },
    {
        "name": "not_equal",
        "args": "(x, y)",
        "category": "arithmetic",
        "description": "Element-wise x != y; returns 1.0 / 0.0",
    },
    # ----- Conditional / state -----
    {
        "name": "if_else",
        "args": "(cond, x, y)",
        "category": "conditional",
        "description": "Element-wise conditional",
    },
    {
        "name": "where",
        "args": "(cond, x, y)",
        "category": "conditional",
        "description": "Alias for if_else with truthy cond",
    },
    {
        "name": "when",
        "args": "(cond, x)",
        "category": "conditional",
        "description": "x where cond is True, else NaN (no carry)",
    },
    {
        "name": "mask",
        "args": "(x, cond)",
        "category": "conditional",
        "description": "Drop cells to NaN where cond is True (inverse of when)",
    },
    {
        "name": "trade_when",
        "args": "(cond, x, exit_cond=None)",
        "category": "conditional",
        "description": "Take x when cond, carry forward; reset on exit_cond",
    },
    {
        "name": "keep",
        "args": "(x, n)",
        "category": "conditional",
        "description": "Per row, keep top-n entries by |x|; zero rest",
    },
    {
        "name": "pasteurize",
        "args": "(x)",
        "category": "conditional",
        "description": "Replace ±inf and NaN with 0",
    },
]

# Rich field metadata: every entry is {name, category, description}.  Returned
# from /api/operators so the frontend can list & autocomplete all 32 fields.
FIELDS: list[dict[str, str]] = [
    # ----- price (7 originals) -----
    {"name": "open", "category": "price", "description": "Opening price"},
    {"name": "high", "category": "price", "description": "Daily high price"},
    {"name": "low", "category": "price", "description": "Daily low price"},
    {"name": "close", "category": "price", "description": "Closing price"},
    {"name": "volume", "category": "price", "description": "Daily share volume"},
    {
        "name": "returns",
        "category": "price",
        "description": "Daily simple return (close.pct_change)",
    },
    {"name": "vwap", "category": "price", "description": "Typical price (high + low + close) / 3"},
    # ----- price structure (7) -----
    {
        "name": "median_price",
        "category": "price_structure",
        "description": "Average of high and low; cleaner price estimate than close",
    },
    {
        "name": "weighted_close",
        "category": "price_structure",
        "description": "Close-weighted typical price; smoother than simple average",
    },
    {
        "name": "range_",
        "category": "price_structure",
        "description": "High minus low; daily volatility proxy (alias: range)",
    },
    {
        "name": "body",
        "category": "price_structure",
        "description": "Absolute candle body size; small body indicates indecision",
    },
    {
        "name": "upper_shadow",
        "category": "price_structure",
        "description": "Rejection of higher prices; potential bearish signal",
    },
    {
        "name": "lower_shadow",
        "category": "price_structure",
        "description": "Rejection of lower prices; potential bullish signal",
    },
    {
        "name": "gap",
        "category": "price_structure",
        "description": "Overnight price gap; captures after-hours sentiment",
    },
    # ----- return variants (5) -----
    {
        "name": "log_returns",
        "category": "return_variants",
        "description": "Log of price ratio; symmetric and better for compounding",
    },
    {
        "name": "abs_returns",
        "category": "return_variants",
        "description": "Absolute daily return; volatility proxy",
    },
    {
        "name": "intraday_return",
        "category": "return_variants",
        "description": "Close minus open over open; pure intraday momentum",
    },
    {
        "name": "overnight_return",
        "category": "return_variants",
        "description": "Open minus prior close; captures news/earnings reaction",
    },
    {
        "name": "signed_volume",
        "category": "return_variants",
        "description": "Volume signed by return direction; money flow proxy",
    },
    # ----- volume & liquidity (4) -----
    {
        "name": "dollar_volume",
        "category": "volume_liquidity",
        "description": "Price times volume; true liquidity measure",
    },
    {
        "name": "adv20",
        "category": "volume_liquidity",
        "description": "20-day average daily volume; liquidity baseline",
    },
    {
        "name": "volume_ratio",
        "category": "volume_liquidity",
        "description": "Volume divided by adv20; values above 1 indicate unusual activity",
    },
    {
        "name": "amihud",
        "category": "volume_liquidity",
        "description": "Absolute return over dollar volume; Amihud illiquidity ratio",
    },
    # ----- volatility & risk (5) -----
    {
        "name": "true_range",
        "category": "volatility_risk",
        "description": "Volatility accounting for gaps; better than simple high-low range",
    },
    {
        "name": "atr",
        "category": "volatility_risk",
        "description": "14-day average true range; standard volatility measure",
    },
    {
        "name": "realized_vol",
        "category": "volatility_risk",
        "description": "20-day rolling return standard deviation",
    },
    {
        "name": "skewness",
        "category": "volatility_risk",
        "description": "60-day rolling return skewness; negative values indicate crash risk",
    },
    {
        "name": "kurtosis",
        "category": "volatility_risk",
        "description": "60-day rolling return kurtosis; high values indicate fat tails",
    },
    # ----- momentum & relative (4) -----
    {"name": "momentum_5", "category": "momentum_relative", "description": "5-day price momentum"},
    {
        "name": "momentum_20",
        "category": "momentum_relative",
        "description": "20-day price momentum",
    },
    {
        "name": "close_to_high_252",
        "category": "momentum_relative",
        "description": "Ratio of close to 52-week high; distance from recent peak",
    },
    {
        "name": "high_low_ratio",
        "category": "momentum_relative",
        "description": "High over low; intraday volatility as a ratio",
    },
    # ----- Phase B: extended momentum (8) -----
    {"name": "momentum_3", "category": "momentum_relative", "description": "3-day price momentum"},
    {
        "name": "momentum_10",
        "category": "momentum_relative",
        "description": "10-day price momentum",
    },
    {
        "name": "momentum_60",
        "category": "momentum_relative",
        "description": "60-day (3-month) price momentum",
    },
    {
        "name": "momentum_120",
        "category": "momentum_relative",
        "description": "120-day (6-month) price momentum",
    },
    {
        "name": "momentum_252",
        "category": "momentum_relative",
        "description": "252-day (1-year) price momentum",
    },
    {
        "name": "reversal_5",
        "category": "momentum_relative",
        "description": "Negative of momentum_5; short-horizon mean-reversion signal",
    },
    {
        "name": "reversal_20",
        "category": "momentum_relative",
        "description": "Negative of momentum_20; mid-horizon mean-reversion signal",
    },
    {
        "name": "momentum_z_60",
        "category": "momentum_relative",
        "description": "60-day momentum / 60-day return-vol; risk-adjusted momentum",
    },
    # ----- Phase B: extended volatility (6) -----
    {
        "name": "realized_vol_5",
        "category": "volatility_risk",
        "description": "5-day rolling return standard deviation",
    },
    {
        "name": "realized_vol_60",
        "category": "volatility_risk",
        "description": "60-day rolling return standard deviation",
    },
    {
        "name": "realized_vol_120",
        "category": "volatility_risk",
        "description": "120-day rolling return standard deviation",
    },
    {
        "name": "vol_of_vol_20",
        "category": "volatility_risk",
        "description": "20-day stdev of realized_vol_20; vol-regime change proxy",
    },
    {
        "name": "parkinson_vol",
        "category": "volatility_risk",
        "description": "Range-based vol estimator using high-low; more efficient than close-to-close",
    },
    {
        "name": "garman_klass_vol",
        "category": "volatility_risk",
        "description": "OHLC-based vol estimator; combines intraday extremes and gap",
    },
    # ----- Phase B: microstructure (8) -----
    {
        "name": "roll_spread",
        "category": "microstructure",
        "description": "Roll's effective spread (1984): 2·sqrt(-cov(Δp_t, Δp_{t-1}))",
    },
    {
        "name": "kyle_lambda",
        "category": "microstructure",
        "description": "Kyle's lambda price-impact proxy: |returns| / sqrt(dollar_volume)",
    },
    {
        "name": "vpin_proxy",
        "category": "microstructure",
        "description": "Order-flow toxicity: |signed_volume| / total_volume, rolling 20",
    },
    {
        "name": "up_volume_ratio",
        "category": "microstructure",
        "description": "Fraction of 20-day volume traded on green days",
    },
    {
        "name": "down_volume_ratio",
        "category": "microstructure",
        "description": "Fraction of 20-day volume traded on red days",
    },
    {
        "name": "turnover_ratio",
        "category": "microstructure",
        "description": "Today's volume vs. 60-day baseline; long-horizon volume_ratio",
    },
    {
        "name": "dollar_amihud",
        "category": "microstructure",
        "description": "Smoothed |returns| / dollar_volume; per-dollar-volume price impact",
    },
    {
        "name": "corwin_schultz",
        "category": "microstructure",
        "description": "Corwin-Schultz (2012) high-low spread estimator",
    },
    # ----- Phase B: extended range / candle structure (6) -----
    {
        "name": "atr_5",
        "category": "volatility_risk",
        "description": "5-day average true range; short-window vol via TR",
    },
    {
        "name": "atr_60",
        "category": "volatility_risk",
        "description": "60-day average true range; long-window vol via TR",
    },
    {
        "name": "range_z_20",
        "category": "price_structure",
        "description": "Z-score of today's range vs its 20-day distribution",
    },
    {
        "name": "body_to_range",
        "category": "price_structure",
        "description": "|close-open| / range; small body = indecision day",
    },
    {
        "name": "consecutive_up",
        "category": "price_structure",
        "description": "Consecutive up-day streak; resets on any down day",
    },
    {
        "name": "consecutive_down",
        "category": "price_structure",
        "description": "Consecutive down-day streak; resets on any up day",
    },
    # ----- Phase C: FRED macro (broadcast to every ticker per day) -----
    {
        "name": "vix",
        "category": "macro",
        "description": "CBOE VIX index — implied vol of S&P 500 options",
    },
    {
        "name": "treasury_3m_yield",
        "category": "macro",
        "description": "3-month Treasury constant-maturity yield (%)",
    },
    {
        "name": "treasury_2y_yield",
        "category": "macro",
        "description": "2-year Treasury constant-maturity yield (%)",
    },
    {
        "name": "treasury_10y_yield",
        "category": "macro",
        "description": "10-year Treasury constant-maturity yield (%)",
    },
    {
        "name": "term_spread_10y_2y",
        "category": "macro",
        "description": "10y minus 2y Treasury spread; recession indicator",
    },
    {
        "name": "term_spread_10y_3m",
        "category": "macro",
        "description": "10y minus 3m Treasury spread; classic curve inversion gauge",
    },
    {
        "name": "high_yield_spread",
        "category": "macro",
        "description": "ICE BofA US High Yield Index option-adjusted spread",
    },
    {
        "name": "baa_yield",
        "category": "macro",
        "description": "Moody's seasoned Baa corporate bond yield (%)",
    },
    {
        "name": "aaa_yield",
        "category": "macro",
        "description": "Moody's seasoned Aaa corporate bond yield (%)",
    },
    {
        "name": "credit_spread_baa_aaa",
        "category": "macro",
        "description": "Baa minus Aaa corporate spread; credit-stress proxy",
    },
    {"name": "dxy", "category": "macro", "description": "Broad trade-weighted USD index"},
    {"name": "wti_oil", "category": "macro", "description": "WTI crude oil spot price (USD/bbl)"},
    # ----- Phase C: yfinance fundamentals (lagged 1 quarter; non-PIT) -----
    {
        "name": "revenue",
        "category": "fundamentals",
        "description": "Quarterly total revenue (lagged 1Q)",
    },
    {"name": "gross_profit", "category": "fundamentals", "description": "Quarterly gross profit"},
    {
        "name": "operating_income",
        "category": "fundamentals",
        "description": "Quarterly operating income",
    },
    {"name": "net_income", "category": "fundamentals", "description": "Quarterly net income"},
    {"name": "ebitda", "category": "fundamentals", "description": "Quarterly EBITDA"},
    {
        "name": "eps",
        "category": "fundamentals",
        "description": "Diluted earnings per share (quarterly)",
    },
    {
        "name": "total_assets",
        "category": "fundamentals",
        "description": "Balance sheet total assets",
    },
    {"name": "total_debt", "category": "fundamentals", "description": "Balance sheet total debt"},
    {
        "name": "total_equity",
        "category": "fundamentals",
        "description": "Balance sheet total stockholders equity (book value)",
    },
    {"name": "cash", "category": "fundamentals", "description": "Cash and cash equivalents"},
    {"name": "current_assets", "category": "fundamentals", "description": "Total current assets"},
    {
        "name": "current_liabilities",
        "category": "fundamentals",
        "description": "Total current liabilities",
    },
    {
        "name": "operating_cash_flow",
        "category": "fundamentals",
        "description": "Cash from operating activities",
    },
    {
        "name": "capex",
        "category": "fundamentals",
        "description": "Capital expenditure (negative in yfinance)",
    },
    {
        "name": "free_cash_flow",
        "category": "fundamentals",
        "description": "OCF + capex (or yfinance-supplied)",
    },
    # Computed ratios (raw fundamentals × close)
    {
        "name": "pe_ratio",
        "category": "fundamentals_ratio",
        "description": "Price-to-earnings: market_cap / net_income",
    },
    {
        "name": "pb_ratio",
        "category": "fundamentals_ratio",
        "description": "Price-to-book: market_cap / total_equity",
    },
    {
        "name": "ps_ratio",
        "category": "fundamentals_ratio",
        "description": "Price-to-sales: market_cap / revenue",
    },
    {
        "name": "ev_ebitda",
        "category": "fundamentals_ratio",
        "description": "Enterprise value / EBITDA",
    },
    {
        "name": "roe",
        "category": "fundamentals_ratio",
        "description": "Return on equity: net_income / total_equity",
    },
    {
        "name": "roa",
        "category": "fundamentals_ratio",
        "description": "Return on assets: net_income / total_assets",
    },
    {
        "name": "debt_to_equity",
        "category": "fundamentals_ratio",
        "description": "Total debt / total equity",
    },
    {
        "name": "current_ratio",
        "category": "fundamentals_ratio",
        "description": "Current assets / current liabilities",
    },
    {
        "name": "gross_margin",
        "category": "fundamentals_ratio",
        "description": "Gross profit / revenue",
    },
    {
        "name": "operating_margin",
        "category": "fundamentals_ratio",
        "description": "Operating income / revenue",
    },
    {
        "name": "fcf_yield",
        "category": "fundamentals_ratio",
        "description": "Free cash flow / market cap",
    },
]

# Plain-name list for callers that only want field names.
DATA_FIELDS: list[str] = [f["name"] for f in FIELDS]
# Sanity: every OHLCV-derived field from the fetcher must be documented in
# FIELDS (macro and GICS fields live alongside but come from other sources).
_ohlcv_documented = set(DATA_FIELDS) & set(ALL_FIELDS)
assert _ohlcv_documented == set(ALL_FIELDS), (
    f"FIELDS missing OHLCV docs for: {set(ALL_FIELDS) - _ohlcv_documented}"
)


# ---------- Lifespan: load data + init DB ----------


_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()

    fetcher = DataFetcher()
    # Load the union of every built-in universe so that switching universes
    # at request time is just a column-filter on already-resolved matrices.
    # Custom universes (Phase 2) lazy-load any tickers missing from this set.
    pool = sorted(universe_all_tickers())
    fetcher.download_universe(tickers=pool)

    spy_fetcher = DataFetcher()
    spy_frames = spy_fetcher.download_universe(tickers=["SPY"], compute_derived=False)
    spy_returns = spy_frames["SPY"]["returns"] if "SPY" in spy_frames else pd.Series(dtype=float)

    # Fama-French 5 factor returns (Mkt-RF, SMB, HML, RMW, CMA + RF) cached
    # weekly.  Network failure is non-fatal — factor decomposition just gets
    # skipped in /api/simulate if the DataFrame is empty.
    ff5 = download_ff5_daily()

    _state["fetcher"] = fetcher
    _state["spy_returns"] = spy_returns
    _state["ff5"] = ff5
    _state["data"] = {field: fetcher.get_data_matrix(field) for field in ALL_FIELDS}
    # GICS-as-DataField wiring: build (dates × tickers) string frames for each
    # GICS level so expressions can reference `sector`, `industry`, etc. and
    # feed them into group_* operators.  Uses the close matrix's index/columns
    # so shapes always align with whatever the user is computing on.
    close_mat = _state["data"].get("close")
    if close_mat is not None and not close_mat.empty:
        _state["gics_data"] = gics_data_frames(close_mat.index, list(close_mat.columns))
    else:
        _state["gics_data"] = {}

    # Phase C — FRED macro fields broadcast to (dates × tickers).  Network
    # failure is non-fatal; missing fields just don't appear in `_state["data"]`
    # and any expression referencing them surfaces a clear "Unknown data field"
    # error from the evaluator.
    macro_present: list[str] = []
    if close_mat is not None and not close_mat.empty:
        macro_series = download_macro()
        ticker_list = list(close_mat.columns)
        for name, series in macro_series.items():
            try:
                _state["data"][name] = macro_broadcast(series, close_mat.index, ticker_list)
                macro_present.append(name)
            except Exception as exc:
                warnings.warn(f"[macro:{name}] broadcast failed: {exc}")
    _state["macro_present"] = macro_present

    # Phase C — yfinance fundamentals.  Slow first-boot (~2 minutes for 95
    # tickers, weekly cache thereafter).  Tickers that fail yfinance contribute
    # NaN columns; the loader never raises.  Wrapped in try/except so a
    # catastrophic yfinance schema break doesn't kill startup.
    fundamentals_present: list[str] = []
    if close_mat is not None and not close_mat.empty:
        try:
            fund_matrices = download_fundamentals(
                tickers=list(close_mat.columns),
                daily_index=close_mat.index,
                close_matrix=close_mat,
            )
            for name, matrix in fund_matrices.items():
                _state["data"][name] = matrix
                fundamentals_present.append(name)
        except Exception as exc:
            warnings.warn(f"[fundamentals] load failed: {exc}; skipping")
    _state["fundamentals_present"] = fundamentals_present
    _state["fundamentals_lag_quarters"] = LAG_QUARTERS
    log.info(
        "lifespan: ready",
        extra={
            "environment": ENVIRONMENT,
            "n_fields": len(_state["data"]),
            "n_tickers": len(fetcher._frames) if hasattr(fetcher, "_frames") else 0,
            "n_universes": len(list_universes()),
            "ff5_present": bool(ff5 is not None and not ff5.empty),
            "auth_enabled": bool(API_TOKEN),
        },
    )
    yield
    log.info("lifespan: shutdown")


app = FastAPI(title="QuantLab", lifespan=lifespan)

# ---------- Rate limiting ----------
# Per-IP limits via slowapi.  Generous in development, tighter in production
# so a single client can't DoS our single-worker free-tier deploy.  Custom 429
# handler keeps the JSON shape consistent with our other errors.
_LIMIT_DEFAULT = "120/minute" if ENVIRONMENT != "production" else "30/minute"
_LIMIT_SIMULATE = "60/minute" if ENVIRONMENT != "production" else "10/minute"
_LIMIT_VALIDATE = "300/minute"  # cheap, debounced from the editor — keep loose
limiter = Limiter(key_func=get_remote_address, default_limits=[_LIMIT_DEFAULT])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# Dev keeps "*" so a browser can hit the API from any localhost variant or
# from a LAN box; production tightens to the configured allow-list.
_cors_origins = ["*"] if ENVIRONMENT != "production" else ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Logging ----------
# Plain text in dev (readable in the uvicorn console), JSON-ish in production
# (one event per line so log aggregators can parse it).  No external deps —
# just stdlib logging configured once at import time.

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Pull through any structured extras attached via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_"):
                continue
            if key in (
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            ):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    if ENVIRONMENT == "production":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    # Replace any handlers uvicorn might have already attached
    root.handlers = [handler]
    root.setLevel(_LOG_LEVEL)
    # Make uvicorn / fastapi loggers respect our config too
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).propagate = False


_configure_logging()
log = logging.getLogger("quantlab")


# ---------- Auth ----------
# Optional bearer-token gate on alpha-mutating endpoints (POST/DELETE on
# /api/alphas).  Set the env var QUANTLAB_API_TOKEN to enable.  When unset,
# auth is bypassed — keeps local dev frictionless.  In production with a
# public URL this prevents drive-by deletion of saved alphas.

API_TOKEN = os.getenv("QUANTLAB_API_TOKEN", "").strip()


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return  # Auth disabled in this deployment
    expected = f"Bearer {API_TOKEN}"
    # `hmac.compare_digest` runs in constant time regardless of where the
    # strings differ, so an attacker can't time-distinguish "first byte wrong"
    # from "almost-right token".  Plain `!=` would be vulnerable in theory.
    provided = authorization or ""
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail=("Missing or invalid Authorization header. Send: Authorization: Bearer <token>"),
        )


# ---------- Helpers ----------


def _resolve_universe(settings: dict) -> tuple[list[str], dict[str, dict[str, str | None]], str]:
    """Pick the ticker list + GICS map from request settings.

    Resolution priority:
        1. ``universe`` (explicit ticker list) — Phase 2 custom universe
        2. ``universe_id`` (built-in preset id)
        3. Default preset id (currently ``sp100_50``)

    For custom universes, lazy-loads any tickers not already in the data pool
    and rebuilds the matrices.  Tickers yfinance can't find are dropped with a
    400 if every supplied ticker fails (the backtest would be empty otherwise).
    """
    custom = settings.get("universe")
    if custom and isinstance(custom, list) and len(custom) > 0:
        tickers = [str(t).upper().strip() for t in custom if str(t).strip()]
        if not tickers:
            raise HTTPException(status_code=400, detail="Custom universe is empty")
        # Lazy-fetch anything new.  Replaces _state["data"] in place since the
        # column set has expanded — the evaluator must see the new tickers.
        fetcher: DataFetcher = _state["fetcher"]
        failed = fetcher.ensure_tickers(tickers)
        # Refresh the matrices snapshot the rest of the request reads from.
        # Rebuild GICS frames + re-broadcast macro to the new (potentially
        # wider) ticker set so the evaluator sees aligned shapes everywhere.
        _state["data"] = {field: fetcher.get_data_matrix(field) for field in ALL_FIELDS}
        close_mat = _state["data"].get("close")
        if close_mat is not None and not close_mat.empty:
            new_cols = list(close_mat.columns)
            _state["gics_data"] = gics_data_frames(close_mat.index, new_cols)
            for name in _state.get("macro_present", []):
                # macro_broadcast wants a Series; pull it out of the existing
                # matrix (any column is fine since they're all identical broadcasts)
                existing = _state["data"].get(name)
                if existing is not None and not existing.empty:
                    series = existing.iloc[:, 0]
                    _state["data"][name] = macro_broadcast(series, close_mat.index, new_cols)
        live = [t for t in tickers if t not in failed]
        if not live:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"None of the {len(tickers)} custom tickers could be loaded "
                    f"from yfinance: {', '.join(failed[:10])}" + ("…" if len(failed) > 10 else "")
                ),
            )
        return live, gics_for(live), "custom"

    uid = settings.get("universe_id") or default_universe_id()
    try:
        u = get_universe(uid)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return u["tickers"], u["gics"], uid


def _make_config(settings: dict | None, *, run_oos: bool = True) -> SimulationConfig:
    s = settings or {}
    tickers, _, _ = _resolve_universe(s)
    return SimulationConfig(
        universe=tickers,
        start_date=s.get("start_date") or DATA_START,
        end_date=s.get("end_date") or DATA_END,
        neutralization=s.get("neutralization", "market"),
        truncation=float(s.get("truncation", 0.05)),
        booksize=float(s.get("booksize", DEFAULT_BOOKSIZE)),
        transaction_cost_bps=float(s.get("transaction_cost_bps", 5.0)),
        decay=int(s.get("decay", 0)),
        oos_split=float(s.get("oos_split", 0.3)),
        run_oos=bool(s.get("run_oos", run_oos)),
        cost_model=s.get("cost_model", "flat"),
        impact_coefficient=float(s.get("impact_coefficient", 0.1)),
        run_walk_forward=bool(s.get("run_walk_forward", False)),
        walk_forward_train_days=int(s.get("walk_forward_train_days", 252)),
        walk_forward_test_days=int(s.get("walk_forward_test_days", 63)),
        walk_forward_step_days=int(s.get("walk_forward_step_days", 63)),
        execution_lag_days=max(1, int(s.get("execution_lag_days", 1))),
        point_in_time_universe=bool(s.get("point_in_time_universe", False)),
    )


def _config_to_dict(cfg: SimulationConfig) -> dict[str, Any]:
    return {
        "universe": cfg.universe,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "neutralization": cfg.neutralization,
        "truncation": cfg.truncation,
        "booksize": cfg.booksize,
        "transaction_cost_bps": cfg.transaction_cost_bps,
        "decay": cfg.decay,
        "oos_split": cfg.oos_split,
        "run_oos": cfg.run_oos,
        "cost_model": cfg.cost_model,
        "impact_coefficient": cfg.impact_coefficient,
        "run_walk_forward": cfg.run_walk_forward,
        "walk_forward_train_days": cfg.walk_forward_train_days,
        "walk_forward_test_days": cfg.walk_forward_test_days,
        "walk_forward_step_days": cfg.walk_forward_step_days,
        "execution_lag_days": cfg.execution_lag_days,
        "point_in_time_universe": cfg.point_in_time_universe,
    }


def _evaluate(expression: str) -> pd.DataFrame:
    # Numeric data fields + GICS string frames live in the same dict so
    # group_* operators can reference `sector`/`industry`/etc. like any other
    # data field.  The dict is read-only inside the evaluator.
    eval_data = {**_state["data"], **_state.get("gics_data", {})}
    try:
        evaluator = AlphaEvaluator(eval_data)
        result = evaluator.evaluate(expression)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression error: {e}")
    if not isinstance(result, pd.DataFrame):
        raise HTTPException(
            status_code=400,
            detail="Expression must produce a (dates × tickers) matrix, not a scalar",
        )
    return result


_METRIC_KEYS = (
    "sharpe",
    "annual_return",
    "annual_vol",
    "max_drawdown",
    "calmar_ratio",
    "sortino_ratio",
    "avg_turnover",
    "fitness",
    "win_rate",
    "profit_factor",
    "beta",
    "information_ratio",
    # Tier 1 signal-quality metrics
    "ic",
    "icir",
    "ic_tstat",
    "ic_pct_positive",
    "ic_n_days",
    "rank_stability",
    "tail_ratio",
    "positive_months_pct",
    "fitness_wq",
)


def _compute_perf_pack(
    result, perf: PerformanceAnalytics, *, spy: pd.Series | None, n_trials: int = 1
):
    """Run analytics on a BacktestResult and shape the (metrics, timeseries) pair."""
    bench = None
    if isinstance(spy, pd.Series) and not spy.empty:
        bench = spy.reindex(pd.to_datetime(result.dates))
    full = perf.compute(result, benchmark_returns=bench, n_trials=n_trials)

    metrics = {k: full[k] for k in _METRIC_KEYS}
    # Pass period info into compare_is_oos through the metrics dict
    metrics["start_date"] = full.get("start_date")
    metrics["end_date"] = full.get("end_date")
    # Per-year breakdown surfaces regime fragility that the headline averages out
    metrics["yearly_returns"] = full.get("yearly_returns", [])
    # Deflated Sharpe carries its own dict (sharpe + p-value + threshold)
    metrics["deflated_sharpe"] = full.get("deflated_sharpe")
    # Alpha-decay carries its own dict (ic_by_horizon + half_life_days + r²)
    metrics["alpha_decay"] = full.get("alpha_decay")
    # Drawdown durations: avg/max/current days underwater
    metrics["drawdown_durations"] = full.get("drawdown_durations")

    timeseries = {
        "dates": list(result.dates),
        "cumulative_pnl": _safe_list(result.cumulative_pnl),
        "daily_returns": _safe_list(result.daily_returns),
        "drawdown": full["drawdown_series"],
        "rolling_sharpe": full["rolling_sharpe"],
        "turnover": _safe_list(result.turnover),
    }
    return metrics, timeseries, full["monthly_returns"]


def _build_response(
    expression: str,
    alpha_matrix: pd.DataFrame,
    cfg: SimulationConfig,
    *,
    n_trials: int = 1,
    gics_map: dict[str, dict[str, str | None]] | None = None,
    universe_id: str | None = None,
) -> dict[str, Any]:
    # Backtester gets the full GICS row per ticker so neutralization can pick
    # any of the four GICS levels.  Falls back to the legacy SECTOR_MAP only
    # for callers that haven't been updated to plumb gics_map through.
    bt = Backtester(_state["data"], sector_map=SECTOR_MAP, gics_map=gics_map)
    try:
        is_result, oos_result = bt.run(alpha_matrix, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Backtest error: {e}")

    spy = _state.get("spy_returns")
    perf = PerformanceAnalytics()

    # Selection bias only applies to IS (the slice the researcher tunes against).
    # OOS is a held-out check — n_trials=1 keeps DSR honest there.
    is_metrics, is_timeseries, is_monthly = _compute_perf_pack(
        is_result, perf, spy=spy, n_trials=n_trials
    )

    oos_metrics: dict[str, Any] | None = None
    oos_timeseries: dict[str, Any] | None = None
    overfitting: dict[str, Any] | None = None
    monthly_returns = is_monthly  # default: only IS data

    if oos_result is not None:
        oos_metrics, oos_timeseries, oos_monthly = _compute_perf_pack(
            oos_result, perf, spy=spy, n_trials=1
        )
        assert oos_metrics is not None  # narrow Optional for mypy
        overfitting = perf.compare_is_oos(is_metrics, oos_metrics)
        # Combined monthly-returns heatmap spans both halves
        monthly_returns = is_monthly + oos_monthly

    # Fama-French 5-factor decomposition on the full backtest window (IS+OOS
    # if both, else just IS).  Tells the user how much of their headline
    # Sharpe is just market-beta + size + value + profitability + investment.
    ff5 = _state.get("ff5")
    decomp_input_dates = list(is_timeseries["dates"])
    decomp_input_returns = list(is_timeseries["daily_returns"])
    if oos_timeseries:
        decomp_input_dates += list(oos_timeseries["dates"])
        decomp_input_returns += list(oos_timeseries["daily_returns"])
    factor_decomp = (
        FactorDecomposition().compute(decomp_input_returns, decomp_input_dates, ff5)
        if ff5 is not None and not ff5.empty
        else None
    )

    # Walk-forward analysis (rolling train→test).  Off by default since it
    # costs ~20× the compute of a single backtest.  When the requested window
    # is too short for even one window the helper returns [] silently.
    walk_forward_windows: list[dict] | None = None
    if cfg.run_walk_forward:
        walk_forward_windows = bt.walk_forward(alpha_matrix, cfg)

    settings_out = _config_to_dict(cfg)
    if universe_id:
        settings_out["universe_id"] = universe_id
    return {
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "is_timeseries": is_timeseries,
        "oos_timeseries": oos_timeseries,
        "overfitting_analysis": overfitting,
        "factor_decomposition": factor_decomp,
        "walk_forward": walk_forward_windows,
        "monthly_returns": monthly_returns,
        "expression": expression,
        "settings": settings_out,
        "data_quality": _data_quality(
            cfg, alpha_matrix, universe_id=universe_id, gics_map=gics_map
        ),
    }


def _data_quality(
    cfg: SimulationConfig,
    alpha_matrix: pd.DataFrame,
    *,
    universe_id: str | None = None,
    gics_map: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Honest disclosure of universe biases that headline metrics inherit."""
    universe_label = universe_id or default_universe_id()
    universe_block: dict[str, Any] = {
        "id": universe_label,
        "ticker_count": len(cfg.universe),
    }
    if gics_map is not None:
        universe_block["available_neutralizations"] = available_neutralizations(gics_map)
        unknown = [t for t, row in gics_map.items() if not row.get("sector")]
        universe_block["tickers_without_gics"] = unknown
    notes = [
        f"Universe in use: {universe_label} ({len(cfg.universe)} tickers). "
        "Each preset is a *current* index snapshot — names that left the "
        "index during the backtest window are absent (survivorship bias). "
        "Estimated Sharpe inflation: 0.1–0.3.",
        "Proper survivorship-free backtests require paid PIT data "
        "(CRSP / Norgate / Sharadar). Documented in README → Drawbacks.",
    ]
    if gics_map is not None and any(not row.get("sector") for row in gics_map.values()):
        notes.append(
            "Some tickers in this custom universe have no GICS classification "
            "in the catalog — sector / industry / sub_industry neutralization "
            "modes will treat them as 'Unknown' and bucket them together."
        )
    pit_block: dict[str, Any] = {"enabled": bool(cfg.point_in_time_universe)}
    if cfg.point_in_time_universe:
        try:
            dates = pd.to_datetime(alpha_matrix.index)
            tickers = list(alpha_matrix.columns)
            summary = membership_summary(pd.DatetimeIndex(dates), tickers)
            pit_block.update(summary)
            if summary["tickers_affected"]:
                affected = ", ".join(
                    f"{a['ticker']} (joined {a['join_date']}, {a['days_masked']} days masked)"
                    for a in summary["tickers_affected"]
                )
                notes.append(
                    "Point-in-time gating ON: " + affected + ". Pre-inclusion "
                    "alpha values were masked to zero so the backtest can't "
                    "anachronistically trade those names."
                )
            else:
                notes.append(
                    "Point-in-time gating ON, but no tickers in this window "
                    "are affected by the curated S&P 100 inclusion-date list."
                )
        except (ValueError, AttributeError):
            pass
    else:
        notes.append(
            "Point-in-time gating OFF — every name in the universe is "
            "treated as tradeable for the entire window. Toggle "
            "'Point-in-time universe' in settings to gate late additions "
            "(currently: TSLA pre-2020-12-21)."
        )

    # Phase C disclosure: fundamentals are non-PIT.  Only surface this if the
    # data was actually loaded — saves the user's eyes when they're using only
    # OHLCV/macro fields.
    fundamentals_present = _state.get("fundamentals_present", [])
    fundamentals_block: dict[str, Any] = {
        "enabled": bool(fundamentals_present),
        "lag_quarters": _state.get("fundamentals_lag_quarters", 0),
        "fields_loaded": len(fundamentals_present),
    }
    if fundamentals_present:
        notes.append(
            f"Fundamentals (P/E, ROE, revenue, …) are lagged "
            f"{_state.get('fundamentals_lag_quarters', 1)} quarter(s) as a "
            "point-in-time proxy: yfinance returns the *latest* filings "
            "(including restatements), so without a lag, alphas using "
            "fundamentals would silently consume future revisions. Real "
            "PIT requires actual report-release dates from a paid feed."
        )
    return {
        "survivorship_bias": True,
        "universe_kind": "current_snapshot",
        "expected_sharpe_inflation": 0.2,
        "universe": universe_block,
        "point_in_time": pit_block,
        "fundamentals": fundamentals_block,
        "notes": notes,
    }


# ---------- Endpoints ----------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/universe")
def get_default_universe():
    """Legacy single-universe endpoint.  Returns the default preset's tickers
    plus the sector map for backwards compatibility with v1 clients.  New
    clients should use /api/universes (plural) for the full preset registry."""
    default = get_universe(default_universe_id())
    sectors = {t: row.get("sector") or "Unknown" for t, row in default["gics"].items()}
    return {"tickers": list(default["tickers"]), "sectors": sectors}


@app.get("/api/examples")
def get_examples():
    """Curated alpha expressions for the Load Example dropdown.

    Each entry is self-contained: the frontend can paste the expression into
    the editor and apply ``recommended_settings`` without further negotiation.
    """
    return {"examples": list_examples()}


@app.get("/api/examples/{example_id}")
def get_example_by_id(example_id: str):
    """Single example lookup — useful if the frontend wants to deep-link
    to a specific alpha (e.g. via URL query param)."""
    e = get_example(example_id)
    if e is None:
        raise HTTPException(status_code=404, detail=f"Unknown example: {example_id!r}")
    return e


@app.get("/api/universes")
def get_universes():
    """All built-in universes + which neutralization modes each one supports.

    The frontend uses this to populate the universe dropdown and disable
    GICS-level neutralization options when the chosen universe doesn't have
    enough distinct groups at that level.
    """
    out = []
    for u in list_universes():
        full = get_universe(u["id"])
        out.append(
            {
                **u,
                "available_neutralizations": available_neutralizations(full["gics"]),
            }
        )
    return {"universes": out, "default": default_universe_id()}


@app.get("/api/operators")
def get_operators():
    return {
        "operators": OPERATORS,
        "fields": FIELDS,
        # Kept for backwards compatibility with any caller that read the old
        # flat-list shape; new code should prefer `fields`.
        "data_fields": DATA_FIELDS,
    }


@app.post("/api/validate")
@limiter.limit(_LIMIT_VALIDATE)
def validate(request: Request, req: ValidateRequest):
    # `request` looks unused but slowapi requires a parameter named exactly
    # "request" to extract the client IP for the per-IP rate-limit key.
    del request
    try:
        ast = Parser().parse(req.expression)
    except ValueError as e:
        return {"valid": False, "error": str(e), "diagnostics": []}

    diagnostics = lint_ast(ast)
    has_lint_errors = any(d["severity"] == "error" for d in diagnostics)
    return {
        "valid": not has_lint_errors,
        "error": (
            next(d["message"] for d in diagnostics if d["severity"] == "error")
            if has_lint_errors
            else None
        ),
        # Full list of warnings + errors so the editor can render them inline.
        "diagnostics": diagnostics,
    }


@app.post("/api/simulate")
@limiter.limit(_LIMIT_SIMULATE)
def simulate(request: Request, req: SimulationRequest):
    del request  # required-by-name for slowapi; not used in the handler
    # Lint before evaluating so we catch look-ahead bias before any
    # inflated-Sharpe number ever leaves the server.
    try:
        ast = Parser().parse(req.expression)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression error: {e}")

    diagnostics = lint_ast(ast)
    errors = [d for d in diagnostics if d["severity"] == "error"]
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"Lint error: {errors[0]['message']}",
        )

    alpha = _evaluate(req.expression)
    # The body's `run_oos` flag wins over anything inside `settings` so
    # callers can toggle the split without rebuilding their settings dict.
    cfg = _make_config(req.settings, run_oos=req.run_oos)
    _, gics_map, universe_id = _resolve_universe(req.settings or {})
    response = _build_response(
        req.expression,
        alpha,
        cfg,
        n_trials=max(1, int(req.n_trials)),
        gics_map=gics_map,
        universe_id=universe_id,
    )
    # Surface any non-fatal lint warnings (long windows, zero shifts) so the
    # frontend can show them next to the dashboard.
    response["diagnostics"] = diagnostics
    log.info(
        "simulate",
        extra={
            "expression": req.expression,
            "is_sharpe": response["is_metrics"].get("sharpe"),
            "oos_sharpe": (response["oos_metrics"] or {}).get("sharpe"),
            "neutralization": cfg.neutralization,
            "run_oos": cfg.run_oos,
            "n_warnings": len([d for d in diagnostics if d["severity"] == "warning"]),
        },
    )
    return response


@app.post("/api/sweep")
@limiter.limit(_LIMIT_SIMULATE)
def sweep(request: Request, req: SweepRequest):
    """Parameter-sweep variant of /api/simulate.

    Expands ``{a..b(:s)?}`` tokens into a cartesian product of expressions,
    runs each through the existing pipeline (IS-only), and returns a flat
    grid of summary metrics the frontend renders as a heatmap or table.
    """
    del request  # required-by-name for slowapi
    try:
        expansion = expand_sweeps(req.expression, max_combinations=req.max_combinations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Sweep error: {e}")

    cfg = _make_config(req.settings, run_oos=False)
    bt = Backtester(_state["data"], SECTOR_MAP)
    perf = PerformanceAnalytics()
    spy = _state.get("spy_returns")

    cells: list[dict[str, Any]] = []
    for i, expr in enumerate(expansion["expressions"]):
        params = combo_for_index(i, expansion)
        try:
            ast = Parser().parse(expr)
            errors = [d for d in lint_ast(ast) if d["severity"] == "error"]
            if errors:
                cells.append(
                    {
                        "expression": expr,
                        "params": params,
                        "error": errors[0]["message"],
                    }
                )
                continue

            alpha = _evaluate(expr)
            is_result, _ = bt.run(alpha, cfg)
            metrics, _, _ = _compute_perf_pack(is_result, perf, spy=spy, n_trials=1)
        except (ValueError, HTTPException) as exc:
            cells.append(
                {
                    "expression": expr,
                    "params": params,
                    "error": str(getattr(exc, "detail", exc)),
                }
            )
            continue

        cells.append(
            {
                "expression": expr,
                "params": params,
                "sharpe": metrics.get("sharpe"),
                "annual_return": metrics.get("annual_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "fitness": metrics.get("fitness"),
                "avg_turnover": metrics.get("avg_turnover"),
            }
        )

    return {
        "expression": req.expression,
        "dimensions": expansion["dimensions"],
        "cells": cells,
        "n_combinations": expansion["total"],
        "settings": _config_to_dict(cfg),
    }


@app.post("/api/compare")
@limiter.limit(_LIMIT_SIMULATE)
def compare_alphas(request: Request, req: CompareRequest):
    """Run 2-4 expressions through the IS-only pipeline and return them all
    overlaid for visual comparison.

    Each expression is evaluated independently (no blending, no shared state).
    The frontend overlays the equity / drawdown / rolling-Sharpe series on the
    same axes and presents the metrics side-by-side.

    No OOS, walk-forward, or factor-decomp here — keeping the response under
    a few seconds even with 4 alphas.  If a user wants validation on a
    winner, they should run it through /api/simulate separately.
    """
    del request  # required-by-name for slowapi

    cfg = _make_config(req.settings, run_oos=False)
    bt = Backtester(_state["data"], SECTOR_MAP)
    perf = PerformanceAnalytics()
    spy = _state.get("spy_returns")

    labels = ["A", "B", "C", "D"]
    out_alphas: list[dict[str, Any]] = []

    for i, expr in enumerate(req.expressions):
        # Per-expression lint — surface look-ahead errors per cell so the user
        # knows which one is bad rather than failing the whole compare run.
        try:
            ast = Parser().parse(expr)
            errors = [d for d in lint_ast(ast) if d["severity"] == "error"]
            if errors:
                out_alphas.append(
                    {
                        "label": labels[i],
                        "expression": expr,
                        "error": errors[0]["message"],
                    }
                )
                continue

            alpha_matrix = _evaluate(expr)
            is_result, _ = bt.run(alpha_matrix, cfg)
            metrics, timeseries, _ = _compute_perf_pack(
                is_result,
                perf,
                spy=spy,
                n_trials=req.n_trials,
            )
        except (ValueError, HTTPException) as exc:
            out_alphas.append(
                {
                    "label": labels[i],
                    "expression": expr,
                    "error": str(getattr(exc, "detail", exc)),
                }
            )
            continue

        out_alphas.append(
            {
                "label": labels[i],
                "expression": expr,
                "metrics": metrics,
                "timeseries": timeseries,
            }
        )

    return {
        "alphas": out_alphas,
        "settings": _config_to_dict(cfg),
    }


@app.post("/api/alphas/multi-blend")
@limiter.limit(_LIMIT_SIMULATE)
def multi_blend(request: Request, req: MultiAlphaRequest):
    del request  # required-by-name for slowapi
    if not req.alphas:
        raise HTTPException(status_code=400, detail="No alphas provided")

    expressions: list[str] = []
    for item in req.alphas:
        expr = item.get("expression")
        if not expr:
            raise HTTPException(status_code=400, detail="Each alpha needs an 'expression'")
        expressions.append(expr)

    # Evaluate every expression once.  We need each alpha's per-day return
    # series for the optimizer (cov-matrix estimation), and we need the alpha
    # matrix itself for the final weighted blend.  Doing both in one pass
    # avoids re-evaluating in two endpoints.
    cfg = _make_config(req.settings)
    _, gics_map, universe_id = _resolve_universe(req.settings or {})
    bt = Backtester(_state["data"], SECTOR_MAP)
    alpha_matrices: list[pd.DataFrame] = []
    return_series: list[pd.Series] = []
    for expr in expressions:
        matrix = _evaluate(expr)
        alpha_matrices.append(matrix)
        # Standalone IS-only run to get the per-day return series for the optimizer.
        # Cheap relative to the full pipeline; OOS isn't needed here.
        try:
            standalone_cfg = _make_config({**(req.settings or {}), "run_oos": False})
            is_result, _ = bt.run(matrix, standalone_cfg)
            return_series.append(
                pd.Series(is_result.daily_returns, index=pd.to_datetime(is_result.dates))
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Standalone backtest failed for {expr!r}: {exc}"
            )

    # Compute weights via the chosen method.  user_weights are only used when
    # weight_method == "equal" — but we treat equal as "user-supplied weights"
    # (the existing semantics) so people can still hand-pick weights.
    if req.weight_method == "equal":
        raw_weights = np.array([float(a.get("weight", 1.0)) for a in req.alphas])
        total = float(np.abs(raw_weights).sum())
        if total == 0:
            raise HTTPException(status_code=400, detail="Weights sum to zero")
        computed = raw_weights / total
        weight_method_used = "equal_user_supplied"
    else:
        try:
            from analytics.mv_optimizer import compute_weights

            returns_df = pd.concat(return_series, axis=1).dropna(how="any")
            if returns_df.empty:
                raise ValueError("All alphas had empty return series after alignment")
            computed = compute_weights(req.weight_method, returns_df, target_vol=req.target_vol)
            weight_method_used = req.weight_method
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Build the weighted blend
    items: list[dict[str, Any]] = []
    combined: pd.DataFrame | None = None
    for w, expr, matrix in zip(computed, expressions, alpha_matrices):
        items.append({"expression": expr, "weight": float(w)})
        weighted = matrix * float(w)
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)

    response = _build_response(
        "multi-blend", combined, cfg, gics_map=gics_map, universe_id=universe_id
    )
    response["expression"] = "multi-blend"
    response["settings"]["alphas"] = items
    response["settings"]["weight_method"] = weight_method_used
    if req.target_vol is not None:
        response["settings"]["target_vol"] = req.target_vol
    return response


@app.post("/api/alphas", dependencies=[Depends(require_api_token)])
async def save_alpha(req: AlphaSaveRequest):
    alpha = _evaluate(req.expression)
    cfg = _make_config(req.settings)
    _, gics_map, universe_id = _resolve_universe(req.settings or {})
    response = _build_response(
        req.expression, alpha, cfg, gics_map=gics_map, universe_id=universe_id
    )
    # The IS/OOS refactor split `metrics` into `is_metrics` + `oos_metrics`;
    # the persisted summary columns track the IS half (the always-present one).
    metrics = response["is_metrics"]
    created_at = datetime.now(timezone.utc).isoformat()

    payload = json.dumps(response, default=str)

    async with connect() as db:
        cursor = await db.execute(
            """
            INSERT INTO alphas
                (name, expression, notes, sharpe, annual_return, max_drawdown,
                 turnover, fitness, created_at, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.name,
                req.expression,
                req.notes,
                metrics.get("sharpe"),
                metrics.get("annual_return"),
                metrics.get("max_drawdown"),
                metrics.get("avg_turnover"),
                metrics.get("fitness"),
                created_at,
                payload,
            ),
        )
        await db.commit()
        row_id = cursor.lastrowid

    log.info(
        "alpha.saved",
        extra={
            # `name` is a reserved LogRecord attribute (the logger name) — use
            # `alpha_name` to avoid the KeyError on Record construction.
            "alpha_id": row_id,
            "alpha_name": req.name,
            "expression": req.expression,
            "sharpe": metrics.get("sharpe"),
        },
    )
    return {
        "id": row_id,
        "name": req.name,
        "expression": req.expression,
        "notes": req.notes,
        "sharpe": metrics.get("sharpe"),
        "created_at": created_at,
    }


@app.get("/api/alphas")
async def list_alphas():
    async with connect() as db:
        db.row_factory = __import__("aiosqlite").Row
        cursor = await db.execute(
            """
            SELECT id, name, expression, notes, sharpe, annual_return,
                   max_drawdown, turnover, fitness, created_at
            FROM alphas
            ORDER BY id DESC
            """
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


@app.get("/api/alphas/{alpha_id}")
async def get_alpha(alpha_id: int):
    async with connect() as db:
        db.row_factory = __import__("aiosqlite").Row
        cursor = await db.execute("SELECT * FROM alphas WHERE id = ?", (alpha_id,))
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Alpha not found")

    record = dict(row)
    raw = record.pop("result_json", None)
    if raw:
        try:
            record["result"] = json.loads(raw)
        except json.JSONDecodeError:
            record["result"] = None
    else:
        record["result"] = None
    return record


@app.delete("/api/alphas/{alpha_id}", dependencies=[Depends(require_api_token)])
async def delete_alpha(alpha_id: int):
    async with connect() as db:
        cursor = await db.execute("DELETE FROM alphas WHERE id = ?", (alpha_id,))
        await db.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alpha not found")
    log.info("alpha.deleted", extra={"alpha_id": alpha_id})
    return {"deleted": alpha_id}


@app.post("/api/alphas/correlations")
async def alphas_correlations(req: CorrelationRequest):
    if not req.alpha_ids:
        raise HTTPException(status_code=400, detail="No alpha_ids provided")

    placeholders = ",".join("?" for _ in req.alpha_ids)
    async with connect() as db:
        db.row_factory = __import__("aiosqlite").Row
        cursor = await db.execute(
            f"SELECT id, name, result_json FROM alphas WHERE id IN ({placeholders})",
            tuple(req.alpha_ids),
        )
        rows = await cursor.fetchall()

    series: dict[str, pd.Series] = {}
    labels: list[str] = []
    for r in rows:
        raw = r["result_json"]
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # New shape uses `is_timeseries` (+ optional `oos_timeseries`).  Old
        # saved alphas (pre-IS/OOS) used a flat `timeseries` key — fall back
        # to that so historical rows still produce correlations.
        is_ts = payload.get("is_timeseries") or {}
        oos_ts = payload.get("oos_timeseries") or {}
        legacy_ts = payload.get("timeseries") or {}

        dates = list(is_ts.get("dates") or legacy_ts.get("dates") or [])
        rets = list(is_ts.get("daily_returns") or legacy_ts.get("daily_returns") or [])
        # If the saved record has both halves, concatenate so the correlation
        # spans the full backtest window.
        if oos_ts.get("dates") and oos_ts.get("daily_returns"):
            dates += list(oos_ts["dates"])
            rets += list(oos_ts["daily_returns"])
        if not dates or not rets:
            continue
        label = r["name"] or f"alpha_{r['id']}"
        # If duplicate names, disambiguate by id
        if label in series:
            label = f"{label}#{r['id']}"
        series[label] = pd.Series(rets, index=pd.to_datetime(dates))
        labels.append(label)

    if not series:
        return {"tickers": [], "matrix": []}

    df = pd.DataFrame(series)
    corr = df.corr().reindex(index=labels, columns=labels)
    matrix = [[_safe_float(v) for v in row] for row in corr.values.tolist()]
    return {"tickers": labels, "matrix": matrix}


@app.get("/api/data/preview")
def data_preview(ticker: str):
    fetcher: DataFetcher = _state["fetcher"]

    # Start with the per-ticker frame (the 7 base fields).
    base = fetcher._frames.get(ticker)
    if base is None:
        path = fetcher._cache_path(ticker)
        if path.exists():
            try:
                base = pd.read_parquet(path)
            except Exception:
                base = None
    if base is None or base.empty:
        raise HTTPException(status_code=404, detail=f"No cached data for {ticker}")

    # Pull this ticker's column out of every derived (dates × tickers) matrix
    # and join it onto the per-ticker frame, so the preview shows all 32 fields.
    columns = {f["name"]: base[f["name"]] for f in FIELDS if f["name"] in base.columns}
    for field, matrix in fetcher._matrix.items():
        if field in columns or matrix is None or matrix.empty:
            continue
        if ticker in matrix.columns:
            columns[field] = matrix[ticker]
    enriched = pd.DataFrame(columns)

    # Order columns to match FIELDS so the response is predictable.
    ordered = [f["name"] for f in FIELDS if f["name"] in enriched.columns]
    enriched = enriched[ordered]

    last30 = enriched.tail(30).copy()
    last30.index = pd.to_datetime(last30.index).strftime("%Y-%m-%d")
    last30 = last30.reset_index().rename(columns={"index": "date"})
    rows: list[dict[str, Any]] = []
    for record in last30.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        rows.append(clean)
    return {"ticker": ticker, "rows": rows}
