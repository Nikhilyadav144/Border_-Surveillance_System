"""Shared fixtures.

Nothing here touches the real ``data/`` directory: every test that needs a
database or a file gets a throwaway one under ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import AppConfig, load_config          # noqa: E402
from core.paths import PROJECT_ROOT                     # noqa: E402
from database.database import Database                  # noqa: E402


@pytest.fixture
def db(tmp_path) -> Database:
    """A fresh SQLite database in a temporary directory."""
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def config() -> AppConfig:
    """The real shipped configuration, loaded and validated."""
    return load_config()


@pytest.fixture
def zone_dicts() -> list[dict]:
    """Two zones: a restricted box in the middle, a border strip on top."""
    return [
        {
            "name": "Border Strip", "kind": "BORDER", "colour": "#ff0000",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.2], [0.0, 0.2]],
        },
        {
            "name": "Restricted Compound", "kind": "RESTRICTED",
            "colour": "#ff8800",
            "polygon": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
        },
        {
            "name": "No Vehicles", "kind": "NO_VEHICLE", "colour": "#ffcc00",
            "polygon": [[0.0, 0.8], [0.3, 0.8], [0.3, 1.0], [0.0, 1.0]],
        },
    ]


@pytest.fixture
def line_dicts() -> list[dict]:
    return [{
        "name": "Forward Boundary",
        "start": [0.0, 0.5], "end": [1.0, 0.5],
        "colour": "#00b4d8", "entering_direction": "north",
    }]


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((540, 960, 3), dtype=np.uint8)


@pytest.fixture
def demo_video() -> Path:
    """The demo clip, skipping the test if it has not been built."""
    path = PROJECT_ROOT / "data" / "input" / "demo_border.mp4"
    if not path.exists():
        pytest.skip("demo clip not built; run scripts/make_demo_video.py")
    return path


def make_box(cx: float, cy: float, w: float = 0.06,
             h: float = 0.16) -> tuple[float, float, float, float]:
    """A normalised box centred on (cx, cy)."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
