"""`alphatest export <id> --format brain` — dump a saved alpha for submission.

Brain's submission accepts a one-line expression plus a metadata header.
The format we emit here is a minimal, copy-pasteable text block that the
researcher can hand to Brain's web UI:

    # Expression for WorldQuant Brain submission
    # Universe:         SP_50_DEFAULT
    # Date range:       2019-01-01 → 2024-12-31
    # In-sample Sharpe: 1.42 (Fitness 0.73)
    # Code signature:   a1b2c3d4e5f6
    # Data signature:   00112233aabb
    # Git hash:         deadbeef
    rank(decay_linear(rank(close), 20))

The intent is human-readable provenance carried alongside the expression
so future-you (and Brain's reviewers) can trace where each submitted
candidate came from.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from db.database import DB_PATH


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "export",
        help="Print a saved alpha in Brain-submission format (text with provenance header).",
    )
    p.add_argument("alpha_id", type=int, help="Saved alpha id (see `alphatest list`).")
    p.add_argument(
        "--format",
        choices=("brain", "json"),
        default="brain",
        help="`brain` = text header + expression; `json` = full saved row.",
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
        print(f"No alphas DB at {db_path}")
        return 1

    row = _load_row(db_path, args.alpha_id)
    if row is None:
        print(f"No alpha with id={args.alpha_id} in {db_path}")
        return 2

    if args.format == "json":
        print(json.dumps(_row_to_dict(row), default=str, indent=2))
        return 0

    _print_brain(row)
    return 0


def _load_row(db_path: Path, alpha_id: int) -> sqlite3.Row | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM alphas WHERE id = ?", (alpha_id,))
        return cur.fetchone()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _print_brain(row: sqlite3.Row) -> None:
    """Brain-style header — `#` comments + the expression as the final line.

    Brain itself doesn't require the comment header (the platform's web UI
    just takes the expression), but printing the provenance alongside makes
    the export self-documenting for the researcher who's choosing what to
    submit out of a long library.
    """
    # Try to dig universe/date range out of the result_json blob.  Older
    # saved alphas might not have a parseable blob — degrade gracefully.
    universe = ""
    date_range = ""
    raw = row["result_json"]
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            cfg = (parsed or {}).get("config") or {}
            uni_list = cfg.get("universe") or []
            universe = f"{len(uni_list)} tickers" if uni_list else ""
            start = cfg.get("start_date") or ""
            end = cfg.get("end_date") or ""
            if start or end:
                date_range = f"{start} → {end}"
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    sharpe = row["sharpe"]
    fitness = row["fitness"]
    sharpe_str = f"{sharpe:+.3f}" if isinstance(sharpe, int | float) else "n/a"
    fitness_str = f"{fitness:.3f}" if isinstance(fitness, int | float) else "n/a"

    print("# Expression for WorldQuant Brain submission")
    print(f"# Saved alpha id: {row['id']}")
    print(f"# Name:           {row['name']}")
    print(f"# Universe:       {universe or '—'}")
    print(f"# Date range:     {date_range or '—'}")
    print(f"# IS Sharpe:      {sharpe_str} (Fitness {fitness_str})")
    print(f"# Code signature: {row['code_signature'] or '—'}")
    print(f"# Data signature: {row['data_signature'] or '—'}")
    print(f"# Git hash:       {row['git_hash'] or '—'}")
    if row["notes"]:
        # Multi-line notes — prefix each line with `#`
        for line in str(row["notes"]).splitlines():
            print(f"# Note:           {line}")
    print()
    print(row["expression"])
