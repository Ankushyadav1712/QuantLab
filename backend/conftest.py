import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def client():
    """Single shared TestClient for the whole test session.

    Triggers the app lifespan once (loading market data from disk cache)
    rather than once-per-module, cutting CI wall-time by ~4x.
    """
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c
