"""Async job queue: the in-memory JobRegistry (unit) and the /api/jobs/sweep
submit → poll → done flow (integration via the session client)."""

import time

import pytest
from engine.jobs import STATUS_DONE, STATUS_QUEUED, JobQueueFull, JobRegistry

NOW = "2026-01-01T00:00:00+00:00"


# ---------- JobRegistry (unit, no clock, no network) ----------


def test_create_and_get():
    reg = JobRegistry()
    job = reg.create("sweep", NOW)
    assert job.status == STATUS_QUEUED
    assert reg.get(job.id) is job
    assert len(job.id) == 12  # uuid4 hex slice


def test_lifecycle_transitions_and_client_view():
    reg = JobRegistry()
    job = reg.create("sweep", NOW)
    reg.set_running(job.id, NOW)
    reg.update_progress(job.id, 3, 10, NOW)
    d = reg.to_dict(reg.get(job.id))
    assert d["status"] == "running"
    assert d["progress"] == {"done": 3, "total": 10}
    assert d["result"] is None  # result withheld until done
    reg.set_done(job.id, {"cells": [1, 2]}, NOW)
    d = reg.to_dict(reg.get(job.id))
    assert d["status"] == STATUS_DONE
    assert d["result"] == {"cells": [1, 2]}


def test_error_state_carries_message():
    reg = JobRegistry()
    job = reg.create("sweep", NOW)
    reg.set_error(job.id, "boom", NOW)
    d = reg.to_dict(reg.get(job.id))
    assert d["status"] == "error"
    assert d["error"] == "boom"
    assert d["result"] is None


def test_progress_ignored_after_terminal():
    reg = JobRegistry()
    job = reg.create("sweep", NOW)
    reg.set_done(job.id, {"ok": True}, NOW)
    reg.update_progress(job.id, 99, 99, NOW)  # must not mutate a finished job
    assert reg.get(job.id).progress_done == 0


def test_max_active_rejects_overflow():
    reg = JobRegistry(max_active=2)
    a = reg.create("sweep", NOW)
    reg.create("sweep", NOW)
    with pytest.raises(JobQueueFull):
        reg.create("sweep", NOW)
    # Finishing a specific job frees exactly one slot.
    reg.set_done(a.id, {}, NOW)
    assert reg.create("sweep", NOW) is not None


def test_list_is_newest_first():
    reg = JobRegistry()
    a = reg.create("sweep", NOW)
    b = reg.create("sweep", NOW)
    assert [j["id"] for j in reg.list()] == [b.id, a.id]


def test_mutations_on_unknown_id_are_noops():
    reg = JobRegistry()
    # None of these should raise on a missing id.
    reg.set_running("nope", NOW)
    reg.update_progress("nope", 1, 2, NOW)
    reg.set_done("nope", {}, NOW)
    reg.set_error("nope", "x", NOW)
    assert reg.get("nope") is None
    assert reg.snapshot("nope") is None


def test_eviction_drops_oldest_terminal_only():
    reg = JobRegistry(max_jobs=2, max_active=10)
    a = reg.create("sweep", NOW)
    b = reg.create("sweep", NOW)
    reg.set_done(a.id, {}, NOW)  # a is terminal, evictable
    c = reg.create("sweep", NOW)  # exceeds max_jobs=2 → evict a
    assert reg.get(a.id) is None
    assert reg.get(b.id) is not None  # active, never evicted
    assert reg.get(c.id) is not None


def test_get_unknown_returns_none():
    assert JobRegistry().get("nope") is None


# ---------- endpoint flow (integration) ----------


def _poll_until_terminal(client, job_id, timeout_s=150):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/jobs/{job_id}").json()
        if last["status"] in ("done", "error"):
            return last
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s; last={last}")


def test_submit_sweep_job_runs_to_completion(client):
    r = client.post(
        "/api/jobs/sweep",
        json={"expression": "rank(close) * {1..3}", "settings": {}, "max_combinations": 10},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"

    final = _poll_until_terminal(client, job_id)
    assert final["status"] == "done", final
    result = final["result"]
    assert result is not None
    assert len(result["cells"]) == 3
    assert result["n_combinations"] == 3
    # Progress reached the total.
    assert final["progress"]["done"] == final["progress"]["total"] == 3


def test_job_result_matches_sync_sweep(client):
    payload = {"expression": "rank(close) * {1..2}", "settings": {}, "max_combinations": 10}
    sync = client.post("/api/sweep", json=payload).json()
    job_id = client.post("/api/jobs/sweep", json=payload).json()["job_id"]
    job = _poll_until_terminal(client, job_id)["result"]
    # Same shape + same number of cells as the synchronous endpoint.
    assert [c["expression"] for c in job["cells"]] == [c["expression"] for c in sync["cells"]]
    assert job["n_combinations"] == sync["n_combinations"]


def test_get_unknown_job_404(client):
    assert client.get("/api/jobs/deadbeef0000").status_code == 404


def test_submit_returns_429_when_queue_full(client, monkeypatch):
    # The only non-200 submit path: JobQueueFull → 429 + Retry-After. Patch
    # create() to raise so no sweep actually runs (fast, deterministic).
    import main
    from engine.jobs import JobQueueFull

    def boom(*a, **k):
        raise JobQueueFull("full")

    monkeypatch.setattr(main._job_registry, "create", boom)
    r = client.post(
        "/api/jobs/sweep",
        json={"expression": "rank(close)", "settings": {}, "max_combinations": 5},
    )
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "10"


def test_bad_expression_job_errors_not_crashes(client):
    r = client.post(
        "/api/jobs/sweep",
        json={"expression": "this is not valid !!!", "settings": {}, "max_combinations": 5},
    )
    assert r.status_code == 200
    final = _poll_until_terminal(client, r.json()["job_id"])
    # A bad sweep expression is a 400 inside the worker → surfaced as job error.
    assert final["status"] == "error"
    assert final["error"]
