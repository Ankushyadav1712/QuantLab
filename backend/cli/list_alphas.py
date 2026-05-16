"""`alphatest list` — print saved alphas as a fixed-width table.

Reads the SQLite database directly (synchronous stdlib `sqlite3`) so the
CLI doesn't drag in aiosqlite + an event loop just to print 50 rows.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

from db.database import DB_PATH


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("list", help="Print saved alphas as a table.")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
        "--order",
        choices=("recent", "sharpe"),
        default="recent",
        help="Sort by save-time (default) or by Sharpe descending.",
    )
    # PDF Section 6.5 — `library --min-sharpe` filter parity.  Implemented as
    # SQL WHERE clauses so a large library still streams efficiently.
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=None,
        help="Only show alphas with Sharpe ≥ N (default: no filter).",
    )
    p.add_argument(
        "--max-dd",
        type=float,
        default=None,
        help="Only show alphas with max drawdown ≥ N (note: drawdowns are negative — pass e.g. -0.25 to exclude > 25%% drawdowns).",
    )
    p.add_argument(
        "--has-provenance",
        action="store_true",
        help="Only show alphas with code+data signatures (saved after the provenance feature shipped).",
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
    if not db_path.exists():
        print(f"No alphas DB at {db_path}.")
        print("Save an alpha through the web UI first, or pass --db <path>.")
        return 1

    rows = _query(
        db_path,
        limit=args.limit,
        order=args.order,
        min_sharpe=args.min_sharpe,
        max_dd=args.max_dd,
        has_provenance=args.has_provenance,
    )
    if not rows:
        # Distinguish "DB is empty" from "filters excluded everything" so the
        # researcher knows whether to widen the filters or save an alpha first.
        any_filter = args.min_sharpe is not None or args.max_dd is not None or args.has_provenance
        if any_filter:
            print("(no alphas match the given filters — try widening them)")
        else:
            print("(no alphas saved yet)")
        return 0

    _print_table(rows)
    return 0


def _query(
    db_path: Path,
    *,
    limit: int,
    order: str,
    min_sharpe: float | None = None,
    max_dd: float | None = None,
    has_provenance: bool = False,
) -> list[sqlite3.Row]:
    """Read up to `limit` alpha rows.  Both `order_by` and the WHERE column
    names are fixed strings (never user-interpolated); the threshold values
    are passed as bound parameters."""
    order_by = "id DESC" if order == "recent" else "sharpe DESC NULLS LAST"
    where_clauses: list[str] = []
    params: list[float | int] = []
    if min_sharpe is not None:
        where_clauses.append("sharpe >= ?")
        params.append(min_sharpe)
    if max_dd is not None:
        where_clauses.append("max_drawdown >= ?")
        params.append(max_dd)
    if has_provenance:
        where_clauses.append("code_signature IS NOT NULL AND data_signature IS NOT NULL")
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(limit)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"""
            SELECT id, name, expression, sharpe, annual_return, max_drawdown,
                   created_at, code_signature, data_signature, git_hash
            FROM alphas
            {where_sql}
            ORDER BY {order_by}
            LIMIT ?
            """,
            params,
        )
        return list(cur.fetchall())


def _print_table(rows: list[sqlite3.Row]) -> None:
    header = (
        f"{'ID':>4}  {'Name':<22}  {'Expression':<32}  "
        f"{'Sharpe':>7}  {'Ret%':>7}  {'DD%':>7}  {'Saved':<10}  {'Code':<7} {'Data':<7}"
    )
    print(header)
    print("─" * len(header))

    for r in rows:
        name = _ellip(r["name"], 22)
        expr = _ellip(r["expression"], 32)
        sharpe = _fmt_num(r["sharpe"])
        ret = _fmt_pct(r["annual_return"])
        dd = _fmt_pct(r["max_drawdown"])
        saved = _fmt_date(r["created_at"])
        code = (r["code_signature"] or "—")[:6]
        data = (r["data_signature"] or "—")[:6]
        print(
            f"{r['id']:>4}  {name:<22}  {expr:<32}  "
            f"{sharpe:>7}  {ret:>7}  {dd:>7}  {saved:<10}  {code:<7} {data:<7}"
        )


def _ellip(s: str | None, width: int) -> str:
    if not s:
        return ""
    return s if len(s) <= width else s[: width - 1] + "…"


def _fmt_num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    # Stored as ISO-8601 with tz suffix; we only show the date part.
    try:
        return datetime.fromisoformat(iso).date().isoformat()
    except ValueError:
        return iso[:10]
