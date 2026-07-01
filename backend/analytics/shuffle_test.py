"""Shuffle leakage test — the empirical "is this alpha real?" check.

Static lint catches obvious look-ahead bugs like ``delay(x, -1)``.  But
subtle leaks — operators that secretly use future values, overfitting on
a small universe, or alphas exploiting an artifact of the data pipeline —
slip past static analysis.  Shuffle test catches them at runtime by
asking the question:

    "If I randomly permute the time order of the input data, does the
    same expression still produce a high Sharpe?"

If yes → the alpha isn't actually predicting next-day returns from
yesterday's signal; it's exploiting some structural property that
survives shuffling (almost always a bug or overfit).

If no — i.e. the real Sharpe is significantly higher than the
shuffle distribution — the alpha has genuine predictive content.

Algorithm:
1. Apply the same row permutation to every data field, preserving each
   day's cross-section but breaking temporal causality between days.
2. Re-evaluate the expression on shuffled data, run the backtest,
   record the Sharpe.
3. Repeat N times.  p-value = ``(count(shuffled >= real) + 1) / (N + 1)``
   (the +1 correction prevents p=0 even when no shuffle beats real).
4. Render a verdict based on p-value AND the shuffle distribution's
   mean — both signals are needed because a perfectly leaky alpha can
   have shuffled-mean ≈ real-mean and high variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Verdict thresholds.  Picked to be conservative-ish: 5% is the standard
# stats convention; the 30% / leakage rules below are heuristic.
_REAL_SIGNAL_P_MAX = 0.05  # p < 5%  → real signal
_BORDERLINE_P_MAX = 0.30  # 5–30%   → borderline
_LEAKAGE_MEAN_SHUFFLED = 0.5  # if even random data Sharpe averages > 0.5,
# the alpha is suspiciously robust to noise


@dataclass
class ShuffleResult:
    """Outcome of a shuffle-leakage test on a single alpha expression."""

    expression: str
    real_sharpe: float
    shuffled_sharpes: list[float] = field(default_factory=list)
    p_value: float | None = None
    percentile: float | None = None
    mean_shuffled: float | None = None
    median_shuffled: float | None = None
    n_shuffles_completed: int = 0
    n_shuffles_failed: int = 0
    verdict: str = "unknown"
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "real_sharpe": _safe(self.real_sharpe),
            "shuffled_sharpes": [_safe(s) for s in self.shuffled_sharpes],
            "p_value": _safe(self.p_value),
            "percentile": _safe(self.percentile),
            "mean_shuffled": _safe(self.mean_shuffled),
            "median_shuffled": _safe(self.median_shuffled),
            "n_shuffles_completed": int(self.n_shuffles_completed),
            "n_shuffles_failed": int(self.n_shuffles_failed),
            "verdict": self.verdict,
            "explanation": self.explanation,
        }


def _safe(x: float | None) -> float | None:
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


def _sharpe_from_returns(daily_returns: list[float] | np.ndarray) -> float | None:
    """Annualised Sharpe from a daily-returns list, NaN-safe."""
    arr = np.asarray(
        [
            x
            for x in daily_returns
            if x is not None and not (isinstance(x, float) and math.isnan(x))
        ],
        dtype=float,
    )
    if arr.size < 2:
        return None
    std = float(arr.std(ddof=1))
    # Use a small epsilon — numpy's std on a numerically-constant array can
    # leak tiny (~1e-19) values from cancellation, blowing up Sharpe to 1e16.
    if std < 1e-12:
        return None
    return float(arr.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _shuffle_data(
    data: dict[str, pd.DataFrame], permutation: np.ndarray
) -> dict[str, pd.DataFrame]:
    """Apply the same row permutation to every (date × ticker) matrix.

    Each row (a day's cross-section) is intact; only the temporal order
    is scrambled.  The DataFrame index is kept in its original
    (chronological) order — every backtest day still has data, it's just
    been sourced from some random *other* day.
    """
    shuffled: dict[str, pd.DataFrame] = {}
    for field_name, matrix in data.items():
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            shuffled[field_name] = matrix
            continue
        # Permutation length might mismatch matrix if the matrix has a
        # different number of rows (e.g. macro fields can be shorter).
        # In that case fall back to permuting just the rows we have.
        n = len(matrix)
        if n != len(permutation):
            local_perm = np.random.default_rng(0).permutation(n)
        else:
            local_perm = permutation
        arr = matrix.to_numpy()
        shuffled_arr = arr[local_perm]
        shuffled[field_name] = pd.DataFrame(
            shuffled_arr, index=matrix.index, columns=matrix.columns
        )
    return shuffled


def run_shuffle_test(
    expression: str,
    *,
    data: dict[str, pd.DataFrame],
    backtester_factory,
    evaluator_factory,
    config,
    n_shuffles: int = 50,
    seed: int = 0,
    progress_callback=None,
) -> ShuffleResult:
    """Run the shuffle leakage test.

    Parameters
    ----------
    expression : str
        The DSL expression to test.
    data : dict[str, pd.DataFrame]
        Field matrices (close, returns, etc.) — same shape used by the
        regular pipeline.
    backtester_factory : callable(data) -> Backtester
        Returns a Backtester wired against the given data dict.  Factory
        rather than instance because we need a fresh one per shuffle
        (avoids accidentally sharing per-instance caches if the
        backtester ever grows them).
    evaluator_factory : callable(data) -> AlphaEvaluator
        Same idea, fresh evaluator per shuffle.  Caller passes their
        ``main._evaluate``-equivalent here.
    config : SimulationConfig
        The simulation config (universe, dates, etc.).
    n_shuffles : int, default 50
        Number of permutation samples.  More → tighter p-value, slower
        run.  Below 30 the p-value resolution is poor; above 100 returns
        diminish.
    seed : int, default 0
        Base seed for the permutation RNG.  Each shuffle uses ``seed + i``.
    progress_callback : callable(i, total) -> None, optional
        Called once per shuffle iteration.  Useful for CLI progress
        printing.

    Returns
    -------
    ShuffleResult with verdict + p-value + distribution stats.
    """
    # 1) Real run
    evaluator = evaluator_factory(data)
    alpha = evaluator.evaluate(expression)
    bt = backtester_factory(data)
    is_result, _ = bt.run(alpha, config)
    real_sharpe = _sharpe_from_returns(is_result.daily_returns)

    result = ShuffleResult(expression=expression, real_sharpe=real_sharpe or float("nan"))
    if real_sharpe is None:
        result.verdict = "real-backtest-failed"
        result.explanation = (
            "The real (unshuffled) backtest produced no usable Sharpe "
            "(zero variance or insufficient observations).  Nothing to compare against."
        )
        return result

    # Take the close matrix as the canonical (date × ticker) shape — every
    # other primary field shares its index.  Permutation length = n_days.
    canonical = data.get("close")
    if canonical is None or canonical.empty:
        result.verdict = "real-backtest-failed"
        result.explanation = "No 'close' matrix in data — cannot derive a permutation length."
        return result
    n_days = len(canonical)

    # 2) Shuffle iterations
    shuffled_sharpes: list[float] = []
    n_failed = 0
    for i in range(int(n_shuffles)):
        rng = np.random.default_rng(seed + i)
        perm = rng.permutation(n_days)
        try:
            sh_data = _shuffle_data(data, perm)
            sh_eval = evaluator_factory(sh_data)
            sh_alpha = sh_eval.evaluate(expression)
            sh_bt = backtester_factory(sh_data)
            sh_result, _ = sh_bt.run(sh_alpha, config)
            sh_sharpe = _sharpe_from_returns(sh_result.daily_returns)
        except (ValueError, RuntimeError):
            sh_sharpe = None
        if sh_sharpe is None:
            n_failed += 1
        else:
            shuffled_sharpes.append(sh_sharpe)
        if progress_callback is not None:
            progress_callback(i + 1, int(n_shuffles))

    result.shuffled_sharpes = shuffled_sharpes
    result.n_shuffles_completed = len(shuffled_sharpes)
    result.n_shuffles_failed = n_failed

    if len(shuffled_sharpes) < 5:
        # Not enough samples to draw any conclusion
        result.verdict = "too-few-shuffles-completed"
        result.explanation = (
            f"Only {len(shuffled_sharpes)} of {n_shuffles} shuffles produced a valid Sharpe. "
            "Try a longer date range or fewer-NaN inputs."
        )
        return result

    arr = np.asarray(shuffled_sharpes)
    n = arr.size
    n_at_least = int((arr >= real_sharpe).sum())
    # One-sample permutation p-value with the standard +1 correction
    p_value = (n_at_least + 1) / (n + 1)
    # Percentile of real within the shuffled distribution (0–100)
    percentile = 100.0 * float((arr < real_sharpe).sum()) / n

    mean_sh = float(arr.mean())
    median_sh = float(np.median(arr))

    # Verdict
    if p_value < _REAL_SIGNAL_P_MAX:
        # Strong p; but if shuffled mean is unusually high too, that's
        # suspicious — both real and shuffled are getting "good" Sharpes,
        # implying the alpha isn't actually time-dependent.
        if mean_sh > _LEAKAGE_MEAN_SHUFFLED:
            verdict = "leakage-suspected"
            explanation = (
                f"Real Sharpe {real_sharpe:.2f} beats {n_at_least}/{n} shuffles "
                f"(p={p_value:.3f}), but shuffled mean is also high ({mean_sh:.2f}). "
                "Even randomly-permuted data produces a strong Sharpe — the alpha "
                "likely exploits a structural artifact, not temporal predictive content."
            )
        else:
            verdict = "real-signal"
            explanation = (
                f"Real Sharpe {real_sharpe:.2f} is at the {percentile:.1f}th percentile of "
                f"shuffled distribution (mean {mean_sh:+.2f}). p={p_value:.3f}. "
                "Predictive content is statistically significant."
            )
    elif p_value < _BORDERLINE_P_MAX:
        verdict = "borderline"
        explanation = (
            f"Real Sharpe {real_sharpe:.2f} at the {percentile:.1f}th percentile of "
            f"shuffled distribution. p={p_value:.3f}. Suggestive but not conclusive — "
            "increase n_shuffles for a tighter estimate, or extend the backtest window."
        )
    else:
        verdict = "indistinguishable-from-noise"
        explanation = (
            f"Real Sharpe {real_sharpe:.2f} sits at the {percentile:.1f}th percentile of "
            f"shuffled distribution (mean {mean_sh:+.2f}). p={p_value:.3f}. "
            "The alpha is statistically indistinguishable from random data — likely overfit "
            "or noise-driven rather than capturing real predictive structure."
        )

    result.p_value = float(p_value)
    result.percentile = float(percentile)
    result.mean_shuffled = mean_sh
    result.median_shuffled = median_sh
    result.verdict = verdict
    result.explanation = explanation
    return result
