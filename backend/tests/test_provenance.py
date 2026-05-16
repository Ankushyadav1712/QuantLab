"""Reproducibility provenance — signature stability + change detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from analytics.provenance import (
    build_provenance,
    compute_code_signature,
    compute_data_signature,
    compute_git_hash,
)

# ---------- compute_code_signature ----------


def test_code_signature_is_deterministic():
    """Same source tree, two hash calls → same digest."""
    a = compute_code_signature()
    b = compute_code_signature()
    assert a == b
    assert len(a) == 12  # 12 hex chars per the contract


def test_code_signature_changes_when_a_tracked_file_changes(tmp_path):
    """Editing any file in the tracked glob must shift the signature."""
    # Mock a minimal backend tree containing just one tracked file.
    (tmp_path / "engine").mkdir()
    target = tmp_path / "engine" / "operators.py"
    target.write_text("def rank(x): return x\n")

    sig_a = compute_code_signature(backend_root=tmp_path)
    target.write_text("def rank(x): return -x\n")  # behaviour-changing edit
    sig_b = compute_code_signature(backend_root=tmp_path)
    assert sig_a != sig_b


def test_code_signature_ignores_untracked_files(tmp_path):
    """Edits to files NOT in _CODE_FILES_GLOB must not affect the sig."""
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "operators.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("doc v1\n")

    sig_before = compute_code_signature(backend_root=tmp_path)
    # Modify an untracked file
    (tmp_path / "README.md").write_text("doc v2 with much more content\n")
    sig_after = compute_code_signature(backend_root=tmp_path)
    assert sig_before == sig_after


def test_code_signature_handles_missing_files(tmp_path):
    """Missing files in the tracked glob shouldn't crash — they just contribute nothing."""
    # tmp_path has none of the tracked files
    sig = compute_code_signature(backend_root=tmp_path)
    assert isinstance(sig, str)
    assert len(sig) == 12


# ---------- compute_data_signature ----------


def test_data_signature_returns_none_for_empty():
    assert compute_data_signature(None) is None
    assert compute_data_signature(pd.DataFrame()) is None


def test_data_signature_deterministic():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(rng.standard_normal((100, 5)), index=dates, columns=list("ABCDE"))
    a = compute_data_signature(df)
    b = compute_data_signature(df)
    assert a == b
    assert isinstance(a, str)
    assert len(a) == 12


def test_data_signature_changes_when_values_change():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame(rng.standard_normal((10, 3)), index=dates, columns=list("ABC"))
    sig_a = compute_data_signature(df)

    df2 = df.copy()
    df2.iloc[0, 0] += 0.001  # tiny change
    sig_b = compute_data_signature(df2)
    assert sig_a != sig_b


def test_data_signature_changes_when_shape_changes():
    rng = np.random.default_rng(0)
    df_small = pd.DataFrame(
        rng.standard_normal((10, 3)),
        index=pd.date_range("2024-01-01", periods=10, freq="B"),
        columns=list("ABC"),
    )
    df_big = pd.DataFrame(
        rng.standard_normal((20, 3)),
        index=pd.date_range("2024-01-01", periods=20, freq="B"),
        columns=list("ABC"),
    )
    assert compute_data_signature(df_small) != compute_data_signature(df_big)


# ---------- compute_git_hash ----------


def test_git_hash_returns_string_or_none():
    """Test repo may or may not be a git repo — both outcomes are fine,
    just shouldn't crash."""
    h = compute_git_hash()
    assert h is None or (isinstance(h, str) and len(h) > 0)


def test_git_hash_non_repo_returns_none(tmp_path):
    """Definitely not a git repo → must return None gracefully."""
    h = compute_git_hash(repo_root=tmp_path)
    assert h is None


# ---------- build_provenance ----------


def test_build_provenance_returns_three_keys():
    out = build_provenance(close_matrix=None)
    assert set(out.keys()) == {"code_signature", "data_signature", "git_hash"}
    assert out["data_signature"] is None  # no close matrix


def test_build_provenance_uses_cached_values_when_provided():
    """Lifespan caches code_sig + git_hash so we don't re-hash 15 files per save."""
    out = build_provenance(
        close_matrix=None,
        cached_code_sig="abc123def456",
        cached_git_hash="deadbeef",
    )
    assert out["code_signature"] == "abc123def456"
    assert out["git_hash"] == "deadbeef"


def test_build_provenance_passes_through_data_signature():
    """Data signature is always recomputed (close matrix can change)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((10, 3)),
        index=pd.date_range("2024-01-01", periods=10, freq="B"),
        columns=list("ABC"),
    )
    out = build_provenance(
        close_matrix=df,
        cached_code_sig="frozen",
        cached_git_hash="frozen-git",
    )
    assert out["data_signature"] is not None
    assert len(out["data_signature"]) == 12


# ---------- /api/alphas save → load round-trip ----------


def test_save_alpha_round_trip_includes_provenance():
    """End-to-end: save an alpha, fetch it back, verify the three signature
    columns are populated on both the list endpoint and the by-id endpoint."""
    import os

    os.environ.setdefault("ENVIRONMENT", "development")
    from fastapi.testclient import TestClient
    from main import app

    token = os.environ.get("QUANTLAB_API_TOKEN", "")
    auth_header = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        # Save a simple alpha
        save_r = client.post(
            "/api/alphas",
            json={
                "name": "provenance-rt-test",
                "expression": "rank(close)",
                "notes": "test",
                "settings": {},
            },
            headers=auth_header,
        )
        assert save_r.status_code in (200, 201), save_r.text
        alpha_id = save_r.json().get("id")
        assert alpha_id is not None

        # Verify list endpoint includes signature columns
        list_r = client.get("/api/alphas")
        rows = [r for r in list_r.json() if r["id"] == alpha_id]
        assert len(rows) == 1
        row = rows[0]
        assert "code_signature" in row
        assert "data_signature" in row
        assert "git_hash" in row
        assert row["code_signature"] is not None
        assert row["data_signature"] is not None
        assert len(row["code_signature"]) == 12
        assert len(row["data_signature"]) == 12

        # Verify by-id endpoint also includes them (via SELECT *)
        load_r = client.get(f"/api/alphas/{alpha_id}")
        body = load_r.json()
        assert body["code_signature"] == row["code_signature"]
        assert body["data_signature"] == row["data_signature"]
        assert body["git_hash"] == row["git_hash"]

        # And the result_json mirror has the same provenance dict
        import json as _json

        result = body.get("result")
        if isinstance(result, str):
            result = _json.loads(result)
        provenance = (result or {}).get("provenance") or {}
        assert provenance.get("code_signature") == row["code_signature"]
        assert provenance.get("data_signature") == row["data_signature"]

        # Cleanup — leave the SQLite as we found it
        client.delete(f"/api/alphas/{alpha_id}", headers=auth_header)
