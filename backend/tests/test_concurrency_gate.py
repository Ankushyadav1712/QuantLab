"""Backtest concurrency gate.

The gate serializes memory-heavy backtests (default one at a time) so two
full-universe runs can't double peak RAM and OOM a small single-worker host
(e.g. Render's 512MB free tier). When every slot is taken it sheds load with a
429 instead of tying up a worker thread — these tests lock in both behaviours.

No data fixture needed: the gate is exercised directly, so this file stays fast.
"""

import threading

import main
import pytest
from fastapi import HTTPException


def test_backtest_slot_yields_when_a_slot_is_free():
    # A fresh, fully-available gate enters and exits the block without raising.
    monkey_gate = threading.BoundedSemaphore(1)
    entered = False
    orig = main._backtest_gate
    main._backtest_gate = monkey_gate
    try:
        with main._backtest_slot():
            entered = True
    finally:
        main._backtest_gate = orig
    assert entered


def test_backtest_slot_sheds_load_with_429_when_saturated(monkeypatch):
    # Point the gate at a semaphore whose only slot is already held, and cut the
    # queue timeout so the test doesn't wait the real 90s. monkeypatch restores
    # both afterwards, so this can't leak state into other tests in the session.
    saturated = threading.BoundedSemaphore(1)
    assert saturated.acquire(blocking=False)  # occupy the sole slot
    monkeypatch.setattr(main, "_backtest_gate", saturated)
    monkeypatch.setattr(main, "_BACKTEST_QUEUE_TIMEOUT", 0.05)

    with pytest.raises(HTTPException) as excinfo:
        with main._backtest_slot():
            pass  # never reached — acquire times out first

    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["Retry-After"] == "5"


def test_backtest_slot_releases_on_exception(monkeypatch):
    # An error inside the block must still free the slot (finally), or the gate
    # would leak a permit on every failed backtest and eventually wedge shut.
    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(main, "_backtest_gate", gate)

    with pytest.raises(ValueError):
        with main._backtest_slot():
            raise ValueError("boom")

    # Slot is free again → can be acquired without blocking.
    assert gate.acquire(blocking=False)
