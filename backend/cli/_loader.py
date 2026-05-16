"""Shared data + universe loader for CLI subcommands.

Mirrors the API's lifespan handler so the CLI sees the same matrices,
universes and GICS maps the running server would.  Factored out of
``scripts/shuffle_test.py`` so every subcommand uses one well-tested path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from config import DATA_END, DATA_START, SECTOR_MAP, UNIVERSE
from data.fetcher import ALL_FIELDS, DataFetcher
from data.universes import default_universe_id, get_universe, gics_data_frames, gics_for
from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator


@dataclass
class LoadedContext:
    """All the moving parts a subcommand needs to evaluate + backtest."""

    data: dict[str, pd.DataFrame]
    close_matrix: pd.DataFrame | None
    universe_id: str
    tickers: list[str]
    gics_map: dict[str, dict[str, str | None]]


def load_context(universe_id: str | None = None, *, verbose: bool = True) -> LoadedContext:
    """Build the data + universe context once for a CLI invocation.

    ``universe_id`` falls back to the project default.  ``verbose`` controls
    the "Loading data..." progress line (off in tests).
    """
    fetcher = DataFetcher()
    if verbose:
        print(f"Loading data: {DATA_START} → {DATA_END} ({len(UNIVERSE)} tickers)...")
    t0 = time.time()
    fetcher.download_universe(UNIVERSE, DATA_START, DATA_END, compute_derived=True)
    data: dict[str, pd.DataFrame] = {field: fetcher.get_data_matrix(field) for field in ALL_FIELDS}
    close_mat = data.get("close")
    if close_mat is not None and not close_mat.empty:
        # GICS string matrices live alongside the price matrices in the same
        # dict — the evaluator looks them up by name (`sector`, `industry`...).
        gics_frames = gics_data_frames(close_mat.index, list(close_mat.columns))
        data = {**data, **gics_frames}
    if verbose:
        print(f"  ({time.time() - t0:.1f}s)")

    uid = universe_id or default_universe_id()
    u = get_universe(uid)
    tickers = list(u["tickers"])
    gics_map = gics_for(tickers)

    return LoadedContext(
        data=data,
        close_matrix=close_mat,
        universe_id=uid,
        tickers=tickers,
        gics_map=gics_map,
    )


def make_config(
    tickers: list[str],
    *,
    neutralization: str = "market",
    booksize: float = 20_000_000,
    run_oos: bool = False,
) -> SimulationConfig:
    """Build a SimulationConfig with CLI-friendly defaults.

    ``run_oos=False`` by default: the CLI's headline number is the full-window
    Sharpe, which matches what `verify_alphas.py` used to print and what users
    quote when comparing alphas.
    """
    return SimulationConfig(
        universe=tickers,
        start_date=str(DATA_START),
        end_date=str(DATA_END),
        neutralization=neutralization,  # type: ignore[arg-type]
        booksize=booksize,
        run_oos=run_oos,
    )


def make_evaluator(data: dict[str, pd.DataFrame]) -> AlphaEvaluator:
    return AlphaEvaluator(data)


def make_backtester(
    data: dict[str, pd.DataFrame],
    gics_map: dict[str, dict[str, str | None]],
) -> Backtester:
    return Backtester(data, sector_map=SECTOR_MAP, gics_map=gics_map)
