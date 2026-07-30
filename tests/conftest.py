"""
Shared fixtures.

The API tests deliberately avoid loading the real 52MB model: a stub recognizer
exercises validation, error handling and response shape in milliseconds. Tests
that need genuine inference are marked `slow` and skip when weights are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.inference import DetectedPlate, DetectionOutcome
from app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = PROJECT_ROOT / 'tests' / 'samples'


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: requires the real model weights')


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temp data dir so tests never touch real output."""
    return Settings(
        data_dir=tmp_path / 'data',
        model_path=PROJECT_ROOT / 'model' / 'best.pt',
        max_upload_mb=1,
        retention_hours=24,
    )


class StubRecognizer:
    """Stands in for PlateRecognizer without loading any model."""

    def __init__(self, plates=None, ready=True, engine='paddleocr'):
        self._plates = plates if plates is not None else []
        self._ready = ready
        self._engine = engine
        self.calls: list[dict] = []

    model_loaded = property(lambda self: self._ready)
    ocr_loaded = property(lambda self: self._ready)
    ready = property(lambda self: self._ready)
    ocr_engine = property(lambda self: self._engine)

    def load(self):  # pragma: no cover - never called in tests
        pass

    def close(self):
        pass

    def detect(self, image, *, max_passes=None, annotate=True):
        self.calls.append({'max_passes': max_passes, 'annotate': annotate,
                           'shape': image.shape})
        height, width = image.shape[:2]
        annotated = image.copy() if annotate else image
        return DetectionOutcome(annotated, list(self._plates), width, height, 12.3)


def make_plate(number='MH12AB1234', valid=True, **overrides) -> DetectedPlate:
    reading = {
        'number': number,
        'confidence': 88.0,
        'valid': valid,
        'format': 'standard' if valid else None,
        'state': 'MH' if valid else None,
        'corrections': 0,
        'variant': 'clahe',
        'passes': 1,
        'votes': 1,
    }
    reading.update(overrides)
    return DetectedPlate((10, 20, 210, 90), 83.5, reading)


@pytest.fixture
def client(settings, monkeypatch):
    """TestClient with the model layer stubbed out."""
    stub = StubRecognizer(plates=[make_plate()])

    app = create_app(settings)

    # Replace the recognizer created during lifespan startup.
    import app.main as main_module

    monkeypatch.setattr(main_module, 'PlateRecognizer', lambda _s: stub)

    with TestClient(app) as test_client:
        test_client.stub = stub
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG."""
    import cv2

    image = np.zeros((60, 200, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    ok, buffer = cv2.imencode('.png', image)
    assert ok
    return buffer.tobytes()
