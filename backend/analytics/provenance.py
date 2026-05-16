"""Reproducibility provenance — bind each saved alpha to the *exact* code
and data snapshot that produced it.

Three signatures are captured per save:

- ``code_signature``: deterministic hash of the backend source tree (the
  files that actually produce the Sharpe number).  Changes whenever any
  operator / backtester / evaluator file is edited.
- ``data_signature``: hash of the canonical cache state (close-matrix
  date range + shape).  Changes when yfinance returns new data.
- ``git_hash``: short SHA of the working-tree commit at boot.  ``None``
  if not in a git repo.

The motivation: a user re-running a saved alpha six months from now needs
to be able to tell whether divergent numbers come from a code change,
a data refresh, or true randomness.  Without these signatures, the most
common reproducibility question — "did my code change?" — has no
answer.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

# Files whose contents determine a backtest's numeric output.  The
# signature hashes *just these files* — not everything in the repo — so
# editing docs / tests / configs doesn't churn the provenance.
_CODE_FILES_GLOB: tuple[str, ...] = (
    "engine/operators.py",
    "engine/evaluator.py",
    "engine/parser.py",
    "engine/backtester.py",
    "engine/lint.py",
    "analytics/performance.py",
    "analytics/ic_metrics.py",
    "analytics/factor_decomp.py",
    "analytics/deflated_sharpe.py",
    "analytics/exposure.py",
    "analytics/stress_test.py",
    "data/fetcher.py",
    "data/universes.py",
    "data/sp100_history.py",
    "config.py",
)


def compute_code_signature(backend_root: Path | str | None = None) -> str:
    """Hash a fixed list of source files that determine backtest output.

    SHA-256 over ``sorted(filepath → content)`` so the order of files in
    the directory listing doesn't affect the result.  Returns the first
    12 hex chars (48 bits of entropy — enough for "did anything change?"
    discrimination across a single-developer project).
    """
    root = Path(backend_root) if backend_root else Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for rel in sorted(_CODE_FILES_GLOB):
        path = root / rel
        if not path.exists():
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        # Read in binary so newline / encoding quirks don't shift the hash
        h.update(path.read_bytes())
        h.update(b"\xff")
    return h.hexdigest()[:12]


def compute_data_signature(close_matrix: pd.DataFrame | None) -> str | None:
    """Hash the canonical (close-matrix) data snapshot.

    The close matrix is the universal anchor: every derived field, every
    backtest, every metric ultimately derives from it.  Hashing its
    shape + first/last date + the literal values gives us a "snapshot
    identity" that changes iff yfinance returns different numbers.

    Returns None when no close data is loaded (lifespan still warming up).
    """
    if close_matrix is None or close_matrix.empty:
        return None
    h = hashlib.sha256()
    h.update(f"{close_matrix.shape}".encode())
    h.update(f"{close_matrix.index[0]}->{close_matrix.index[-1]}".encode())
    # Hash the underlying numpy buffer — fast, deterministic, byte-exact.
    # We use .tobytes() rather than iterating values because at 1500×100 it's
    # ~1.2 MB and hashing that takes ~3 ms.
    h.update(close_matrix.to_numpy().tobytes())
    return h.hexdigest()[:12]


def compute_git_hash(repo_root: Path | str | None = None) -> str | None:
    """Short SHA of HEAD, or None if not in a git repo / git not available.

    Doesn't include the dirty-tree marker by design: we already capture
    the code_signature, which catches uncommitted changes more precisely
    than git status would (a comment-only edit would dirty the tree but
    not change the signature).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha if sha else None


def build_provenance(
    *,
    close_matrix: pd.DataFrame | None = None,
    backend_root: Path | str | None = None,
    cached_code_sig: str | None = None,
    cached_git_hash: str | None = None,
) -> dict[str, Any]:
    """Bundle the three signatures into the dict stored alongside an alpha.

    ``cached_code_sig`` / ``cached_git_hash`` let the lifespan handler
    compute these once at startup and reuse them across every save — the
    code + git state don't change without a process restart, so there's
    no reason to re-hash 15 source files on every save.
    """
    return {
        "code_signature": cached_code_sig or compute_code_signature(backend_root),
        "data_signature": compute_data_signature(close_matrix),
        "git_hash": cached_git_hash if cached_git_hash is not None else compute_git_hash(),
    }
