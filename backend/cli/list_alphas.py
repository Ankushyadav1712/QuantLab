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

    rows = _query(db_path, limit=args.limit, order=args.order)
    if not rows:
        print("(no alphas saved yet)")
        return 0

    _print_table(rows)
    return 0


def _query(db_path: Path, *, limit: int, order: str) -> list[sqlite3.Row]:
    """Read up to `limit` alpha rows.  `order_by` is fixed-string SQL, never
    interpolated from user input."""
    order_by = "id DESC" if order == "recent" else "sharpe DESC NULLS LAST"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"""
            SELECT id, name, expression, sharpe, annual_return, max_drawdown,
                   created_at, code_signature, data_signature, git_hash
            FROM alphas
            ORDER BY {order_by}
            LIMIT ?
            """,
            (limit,),
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
