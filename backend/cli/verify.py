"""`alphatest verify <id>` — re-run a saved alpha and check provenance.

The point: when a saved alpha is six months old, the user wants to know
"if I re-run this today, do I get the same numbers?"  This subcommand
answers that by:

1. Loading the saved row (expression + stored signatures + headline Sharpe)
2. Re-running the backtest with the current code/data
3. Computing fresh signatures
4. Diffing — and explaining what each diff *means*

Exit code is 0 only when the headline Sharpe matches within tolerance.
Signature mismatches always print a warning but don't fail (a code edit
may be intentional).
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from analytics.performance import PerformanceAnalytics
from analytics.provenance import build_provenance
from db.database import DB_PATH

from cli._loader import LoadedContext, load_context, make_backtester, make_config, make_evaluator

# Sharpe tolerance — a saved alpha's headline matches if |Δ| ≤ this.
# Set to 1e-3 (the precision we print to) so floating-point chatter doesn't
# trigger a false alarm but a genuine logic change does.
DEFAULT_SHARPE_TOL = 1e-3


@dataclass
class StoredAlpha:
    id: int
    name: str
    expression: str
    sharpe: float | None
    code_signature: str | None
    data_signature: str | None
    git_hash: str | None


@dataclass
class VerifyOutcome:
    """Comparison result.  ``ok`` is the single-bit verdict the CLI returns
    as its exit code; the rest is for the human-readable report."""

    ok: bool
    stored: StoredAlpha
    fresh_sharpe: float | None
    fresh_code_signature: str | None
    fresh_data_signature: str | None
    fresh_git_hash: str | None
    sharpe_delta: float | None  # None when either side is None
    code_changed: bool
    data_changed: bool
    git_changed: bool


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "verify",
        help="Re-run a saved alpha and check the headline + provenance still match.",
    )
    p.add_argument("alpha_id", type=int, help="ID of the saved alpha (see `alphatest list`).")
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_SHARPE_TOL,
        help=f"Sharpe-match tolerance (default {DEFAULT_SHARPE_TOL}).",
    )
    p.add_argument(
        "--db",
        default=None,
        help="Override the SQLite path (defaults to backend/data/quantlab.db).",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    db_path = Path(args.db) if args.db else DB_PATH
    stored = _load_alpha(db_path, args.alpha_id)
    if stored is None:
        print(f"No alpha with id={args.alpha_id} in {db_path}")
        return 2

    print(f"Verifying alpha #{stored.id}: {stored.name!r}")
    print(f"  Expression: {stored.expression}")
    print(f"  Stored Sharpe:  {_fmt(stored.sharpe)}")
    print(f"  Stored code sig: {stored.code_signature or '—'}")
    print(f"  Stored data sig: {stored.data_signature or '—'}")
    print(f"  Stored git hash: {stored.git_hash or '—'}\n")

    ctx = load_context(verbose=True)
    outcome = verify(stored, ctx, tolerance=args.tolerance)
    _print_outcome(outcome, tolerance=args.tolerance)
    return 0 if outcome.ok else 1


def _load_alpha(db_path: Path, alpha_id: int) -> StoredAlpha | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id, name, expression, sharpe,
                   code_signature, data_signature, git_hash
            FROM alphas WHERE id = ?
            """,
            (alpha_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return StoredAlpha(
        id=row["id"],
        name=row["name"],
        expression=row["expression"],
        sharpe=row["sharpe"],
        code_signature=row["code_signature"],
        data_signature=row["data_signature"],
        git_hash=row["git_hash"],
    )


def verify(
    stored: StoredAlpha,
    ctx: LoadedContext,
    *,
    tolerance: float = DEFAULT_SHARPE_TOL,
) -> VerifyOutcome:
    """Re-run `stored.expression` against `ctx` and diff against the stored row.

    Pure function: takes loaded inputs, returns a VerifyOutcome.  Splitting
    this from `handle()` keeps it easy to unit-test without spinning up the
    whole CLI process.
    """
    cfg = make_config(ctx.tickers)
    evaluator = make_evaluator(ctx.data)
    backtester = make_backtester(ctx.data, ctx.gics_map)
    perf = PerformanceAnalytics()

    fresh_sharpe: float | None = None
    try:
        alpha = evaluator.evaluate(stored.expression)
        is_result, _oos = backtester.run(alpha, cfg)
        metrics = perf.compute(is_result, gics_map=ctx.gics_map)
        fresh_sharpe = metrics.get("sharpe")
    except Exception as exc:  # noqa: BLE001
        # We still want to print the signature diff even if the backtest
        # blew up — that's diagnostic value too.
        print(f"\n  ⚠ Backtest failed during verify: {exc}\n")

    fresh_prov = build_provenance(close_matrix=ctx.close_matrix)

    sharpe_delta = (
        fresh_sharpe - stored.sharpe
        if (fresh_sharpe is not None and stored.sharpe is not None)
        else None
    )
    ok = sharpe_delta is not None and abs(sharpe_delta) <= tolerance

    return VerifyOutcome(
        ok=ok,
        stored=stored,
        fresh_sharpe=fresh_sharpe,
        fresh_code_signature=fresh_prov.get("code_signature"),
        fresh_data_signature=fresh_prov.get("data_signature"),
        fresh_git_hash=fresh_prov.get("git_hash"),
        sharpe_delta=sharpe_delta,
        code_changed=_changed(stored.code_signature, fresh_prov.get("code_signature")),
        data_changed=_changed(stored.data_signature, fresh_prov.get("data_signature")),
        git_changed=_changed(stored.git_hash, fresh_prov.get("git_hash")),
    )


def _changed(stored: str | None, fresh: str | None) -> bool:
    """A signature counts as changed only when both sides are present and
    differ.  Missing-on-one-side is "unknown", not "changed", so old alphas
    saved before the provenance feature shipped don't trigger spurious
    warnings."""
    if stored is None or fresh is None:
        return False
    return stored != fresh


def _print_outcome(o: VerifyOutcome, *, tolerance: float) -> None:
    print("── RESULTS " + "─" * 50)
    print(f"  Fresh Sharpe:  {_fmt(o.fresh_sharpe)}")
    if o.sharpe_delta is None:
        print("  Sharpe match:  unknown (either side missing)")
    else:
        marker = "✓" if o.ok else "✗"
        print(f"  Sharpe match:  {marker} Δ = {o.sharpe_delta:+.4f} (tol {tolerance})")

    print(f"  Fresh code sig: {o.fresh_code_signature or '—'}"
          + (" (changed)" if o.code_changed else ""))
    print(f"  Fresh data sig: {o.fresh_data_signature or '—'}"
          + (" (changed)" if o.data_changed else ""))
    print(f"  Fresh git hash: {o.fresh_git_hash or '—'}"
          + (" (changed)" if o.git_changed else ""))

    # Diagnostic story.  When all three are stable and Sharpe matches, you
    # have full reproducibility.  When one changes, we name the likely cause.
    print()
    if o.ok and not (o.code_changed or o.data_changed):
        print("  ✓ Reproduced byte-identically.")
    elif o.ok:
        print("  ✓ Headline reproduced, but signatures shifted — verify the diff is intentional.")
    else:
        causes = []
        if o.code_changed:
            causes.append("backend code edits since save")
        if o.data_changed:
            causes.append("data refresh (yfinance returned different numbers)")
        if o.git_changed and not o.code_changed:
            causes.append("git HEAD moved (but tracked source files didn't)")
        if not causes:
            # Distinguish "we have no idea why" from "the alpha pre-dates the
            # provenance feature so the absence of a stored sig just means
            # we can't tell — not that the engine is non-deterministic".
            if (
                o.stored.code_signature is None
                and o.stored.data_signature is None
                and o.stored.git_hash is None
            ):
                causes.append(
                    "alpha was saved before provenance tracking shipped — "
                    "code/data state at save time is unknown, so drift is expected"
                )
            else:
                causes.append("non-determinism inside the engine — investigate")
        print("  ✗ Headline drifted.  Likely cause(s): " + "; ".join(causes) + ".")
    print("─" * 60)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.4f}"
