"""In-process async job registry for long-running backtests (sweeps).

Why: a big parameter sweep runs many backtests and can exceed the HTTP request
timeout (and blocks the browser waiting).  Submitting it as a *job* returns a
``job_id`` immediately; the work runs in the threadpool (never on the event
loop) and the client polls ``GET /api/jobs/{id}`` for progress + result.

Free-tier scope: the store is in-memory (lost on restart) and there's no
separate worker process — execution is still serialized by the existing
backtest gate, so this buys non-blocking submission + progress, not
parallelism.  Redis / a worker service would add durability + true
concurrency later.

Thread-safety: the registry is mutated from both the event loop (create) and
threadpool worker threads (progress/done/error), so every access takes the
lock.  Timestamps are injected by the caller (the module avoids importing a
clock so it stays trivially testable).
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

# Job lifecycle: queued → running → (done | error).
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
_TERMINAL = (STATUS_DONE, STATUS_ERROR)


class JobQueueFull(Exception):
    """Raised when too many jobs are active (queued/running) to accept another."""


@dataclass
class Job:
    id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    progress_done: int = 0
    progress_total: int = 0
    result: Any = None
    error: str | None = None
    _extra: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL


class JobRegistry:
    def __init__(self, max_jobs: int = 100, max_active: int = 8) -> None:
        # Insertion-ordered so eviction can drop the oldest *terminal* job.
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        self._max_active = max_active

    # ---- creation -----------------------------------------------------
    def create(self, kind: str, now: str, *, extra: dict[str, Any] | None = None) -> Job:
        """Register a new queued job.  Raises JobQueueFull if too many are active."""
        with self._lock:
            active = sum(1 for j in self._jobs.values() if not j.is_terminal())
            if active >= self._max_active:
                raise JobQueueFull(f"{active} jobs already active (max {self._max_active})")
            job = Job(
                id=uuid.uuid4().hex[:12],
                kind=kind,
                status=STATUS_QUEUED,
                created_at=now,
                updated_at=now,
                _extra=dict(extra or {}),
            )
            self._jobs[job.id] = job
            self._evict_locked()
            return job

    def _evict_locked(self) -> None:
        # Drop oldest terminal jobs until under the cap; never evict an active
        # job (it's still running / results not yet fetched).
        while len(self._jobs) > self._max_jobs:
            victim = next((jid for jid, j in self._jobs.items() if j.is_terminal()), None)
            if victim is None:
                break
            del self._jobs[victim]

    # ---- mutation (called from worker threads) ------------------------
    def set_running(self, job_id: str, now: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == STATUS_QUEUED:
                job.status = STATUS_RUNNING
                job.updated_at = now

    def update_progress(self, job_id: str, done: int, total: int, now: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and not job.is_terminal():
                job.progress_done = done
                job.progress_total = total
                job.updated_at = now

    def set_done(self, job_id: str, result: Any, now: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = STATUS_DONE
                job.result = result
                job.updated_at = now

    def set_error(self, job_id: str, error: str, now: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = STATUS_ERROR
                job.error = error
                job.updated_at = now

    # ---- reads --------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        """Consistent client-facing view of one job, or None if absent.

        Reads all fields under the lock so a poll can't observe a torn state
        (e.g. status still 'running' while the worker has already assigned the
        result).  Result is only inlined once terminal — a sweep result is
        large; polling shouldn't ship it on every tick."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._to_dict_locked(job)

    def _to_dict_locked(self, job: Job) -> dict[str, Any]:
        d = asdict(job)
        d.pop("_extra", None)
        d["progress"] = {"done": d.pop("progress_done"), "total": d.pop("progress_total")}
        if job.status != STATUS_DONE:
            d["result"] = None
        return d

    # Retained for unit tests that build a view from a Job directly; wraps the
    # locked reader so field reads are still synchronized against workers.
    def to_dict(self, job: Job) -> dict[str, Any]:
        with self._lock:
            return self._to_dict_locked(job)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())[-limit:][::-1]
            # Build summaries INSIDE the lock so per-job field reads are
            # synchronized against worker-thread mutation.  Omit the result.
            return [
                {
                    "id": j.id,
                    "kind": j.kind,
                    "status": j.status,
                    "created_at": j.created_at,
                    "progress": {"done": j.progress_done, "total": j.progress_total},
                }
                for j in jobs
            ]
