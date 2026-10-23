import os
import numpy as np
import pytest

from anpr.config import load_config
from anpr.aggregation import FinalizedDetection

_EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "config.yaml.example")


@pytest.fixture(autouse=True)
def _no_real_redis(monkeypatch):
    """Run tests against the in-memory / local-file fallbacks, never a real Redis on the
    box. Keeps each test isolated (the tracker + queue would otherwise share live Redis
    state across tests) and avoids touching a live Redis DB.
    """
    monkeypatch.setattr("anpr.storage_redis._redis", None)
    monkeypatch.setattr("anpr.tracker._redis", None)
    # keep per-point/camera log FILES out of the repo during tests (use the console only)
    monkeypatch.setenv("ANPR_DISABLE_FILE_LOGS", "1")


@pytest.fixture
def cfg(tmp_path):
    c = load_config(_EXAMPLE)
    c.inference.engine = "mock"
    c.dry_run = True
    c.sqlite.path = str(tmp_path / "anpr.db")
    c.doppelganger_path = str(tmp_path / "dop.csv")
    c.whitelist_path = str(tmp_path / "missing_whitelist.yaml")  # absent -> disabled
    return c


@pytest.fixture
def img():
    return (np.random.rand(96, 192, 3) * 255).astype(np.uint8)


@pytest.fixture
def frame():
    return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)


@pytest.fixture
def make_finalized():
    """Factory for FinalizedDetection. A fixture (not an importable helper) so tests never
    need `from tests.conftest import ...` — that name collides with ultralytics' own
    top-level `tests` package once the ML deps are installed."""
    def _make(text, point, ts, img, frame, camera_id="gatein_cam1"):
        return FinalizedDetection(camera_id, point, text, [text], img, frame, 0.92, ts, 5)
    return _make
