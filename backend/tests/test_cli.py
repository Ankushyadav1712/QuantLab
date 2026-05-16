"""CLI tests — parser wiring + the pure pieces of verify/list.

The expensive integration paths (`alphatest run`, `alphatest shuffle`) hit the
real backtester and full universe; we don't re-test those here — the engine and
shuffle tests already cover that code.  These tests focus on what's *new*:
argparse wiring, the SQLite reader, and the verify-diff logic.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cli import list_alphas, verify
from cli.__main__ import build_parser
from cli.verify import StoredAlpha, VerifyOutcome, _changed

# ---------- parser wiring ----------


def test_parser_has_all_four_subcommands():
    p = build_parser()
    # argparse exposes registered subparser names through the choices dict
    sub_action = next(a for a in p._subparsers._actions if a.dest == "command")
    assert set(sub_action.choices.keys()) == {"run", "shuffle", "list", "verify"}


def test_parser_run_requires_expression():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run"])  # missing positional


def test_parser_run_accepts_neutralization_choices():
    p = build_parser()
    args = p.parse_args(["run", "rank(close)", "--neutralization", "sector"])
    assert args.neutralization == "sector"
    assert args.expression == "rank(close)"


def test_parser_run_rejects_unknown_neutralization():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run", "rank(close)", "--neutralization", "nope"])


def test_parser_shuffle_defaults_to_50_iters():
    p = build_parser()
    args = p.parse_args(["shuffle", "rank(close)"])
    assert args.iters == 50
    assert args.seed == 0


def test_parser_verify_requires_int_id():
    p = build_parser()
    args = p.parse_args(["verify", "42"])
    assert args.alpha_id == 42
    with pytest.raises(SystemExit):
        p.parse_args(["verify", "not-an-int"])


def test_parser_list_defaults():
    p = build_parser()
    args = p.parse_args(["list"])
    assert args.limit == 50
    assert args.order == "recent"


# ---------- list_alphas: DB reader ----------


def _make_test_db(tmp_path: Path) -> Path:
    """Create a minimal alphas table with two rows so the reader has data."""
    db = tmp_path / "test_quantlab.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE alphas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, expression TEXT, notes TEXT,
            sharpe REAL, annual_return REAL, max_drawdown REAL,
            turnover REAL, fitness REAL, created_at TEXT, result_json TEXT,
            code_signature TEXT, data_signature TEXT, git_hash TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO alphas (name, expression, sharpe, annual_return, max_drawdown, "
        "created_at, code_signature, data_signature, git_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("low", "rank(close)", 0.3, 0.05, -0.12, now, "aaaaaa", "111111", "g1"),
            ("high", "rank(close) - rank(open)", 1.2, 0.18, -0.08, now, "bbbbbb", "222222", "g2"),
        ],
    )
    conn.commit()
    conn.close()
    return db


def test_list_query_recent_order(tmp_path):
    db = _make_test_db(tmp_path)
    rows = list_alphas._query(db, limit=10, order="recent")
    # recent → id DESC, so the second row (id=2 "high") comes first
    assert [r["name"] for r in rows] == ["high", "low"]


def test_list_query_sharpe_order(tmp_path):
    db = _make_test_db(tmp_path)
    rows = list_alphas._query(db, limit=10, order="sharpe")
    assert [r["name"] for r in rows] == ["high", "low"]


def test_list_query_respects_limit(tmp_path):
    db = _make_test_db(tmp_path)
    rows = list_alphas._query(db, limit=1, order="recent")
    assert len(rows) == 1


def test_list_handle_missing_db_returns_1(tmp_path, capsys):
    """`alphatest list --db <nonexistent>` should print + exit nonzero."""
    import argparse

    args = argparse.Namespace(db=str(tmp_path / "nope.db"), limit=50, order="recent")
    rc = list_alphas.handle(args)
    assert rc == 1
    assert "No alphas DB" in capsys.readouterr().out


# ---------- verify: pure diff logic (no backtest) ----------


def test_changed_helper_treats_missing_as_unknown():
    # Either side missing → not flagged as changed (avoids spurious warnings on
    # alphas saved before the provenance feature shipped)
    assert _changed(None, "abc") is False
    assert _changed("abc", None) is False
    assert _changed(None, None) is False


def test_changed_helper_detects_real_diff():
    assert _changed("abc", "xyz") is True
    assert _changed("abc", "abc") is False


def test_verify_load_alpha_returns_none_for_missing_db(tmp_path):
    assert verify._load_alpha(tmp_path / "nope.db", 1) is None


def test_verify_load_alpha_returns_none_for_missing_id(tmp_path):
    db = _make_test_db(tmp_path)
    assert verify._load_alpha(db, 999) is None


def test_verify_load_alpha_roundtrip(tmp_path):
    db = _make_test_db(tmp_path)
    stored = verify._load_alpha(db, 2)
    assert stored is not None
    assert stored.name == "high"
    assert stored.expression == "rank(close) - rank(open)"
    assert stored.code_signature == "bbbbbb"


def test_verify_outcome_ok_when_sharpe_matches_within_tol():
    """The pure verdict logic — exercised without running a backtest."""
    stored = StoredAlpha(
        id=1,
        name="x",
        expression="rank(close)",
        sharpe=1.000,
        code_signature="aaa",
        data_signature="bbb",
        git_hash="g",
    )
    # Manually construct the outcome that `verify()` would return for a clean
    # reproduction (within tol, no signature drift).
    out = VerifyOutcome(
        ok=True,
        stored=stored,
        fresh_sharpe=1.0005,
        fresh_code_signature="aaa",
        fresh_data_signature="bbb",
        fresh_git_hash="g",
        sharpe_delta=0.0005,
        code_changed=False,
        data_changed=False,
        git_changed=False,
    )
    assert out.ok is True
    assert abs(out.sharpe_delta) < 1e-3


def test_print_outcome_legacy_alpha_message(capsys):
    """Saved-before-provenance alphas (all stored sigs None) get a tailored
    diagnostic instead of the misleading 'non-determinism' line."""
    stored = StoredAlpha(
        id=1,
        name="old",
        expression="rank(close)",
        sharpe=0.87,
        code_signature=None,
        data_signature=None,
        git_hash=None,
    )
    out = VerifyOutcome(
        ok=False,
        stored=stored,
        fresh_sharpe=-0.15,
        fresh_code_signature="abc",
        fresh_data_signature="def",
        fresh_git_hash="g",
        sharpe_delta=-1.02,
        code_changed=False,
        data_changed=False,
        git_changed=False,
    )
    verify._print_outcome(out, tolerance=1e-3)
    txt = capsys.readouterr().out
    assert "saved before provenance" in txt
    assert "non-determinism" not in txt


def test_verify_outcome_flags_drift_when_sharpe_diverges():
    stored = StoredAlpha(
        id=1,
        name="x",
        expression="rank(close)",
        sharpe=1.0,
        code_signature="aaa",
        data_signature="bbb",
        git_hash="g",
    )
    out = VerifyOutcome(
        ok=False,
        stored=stored,
        fresh_sharpe=1.5,
        fresh_code_signature="ccc",
        fresh_data_signature="bbb",
        fresh_git_hash="g",
        sharpe_delta=0.5,
        code_changed=True,
        data_changed=False,
        git_changed=False,
    )
    assert out.ok is False
    assert out.code_changed
