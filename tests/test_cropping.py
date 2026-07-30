"""
Crop padding around a detected plate.

A fixed 10px margin clipped the trailing character of a Bharat-series plate:
the detector box hugs the glyphs, so "22BH6517A" was cropped mid-"A" and two of
five preprocessing variants read "22BH6517" instead. Padding is therefore
proportional to the box.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.inference import PlateRecognizer


class OneBoxModel:
    """Minimal stand-in for a YOLO result exposing a single box."""

    def __init__(self, box, conf=0.9):
        self._box = box
        self._conf = conf

    def predict(self, source, conf, verbose):        # noqa: A002
        box = self._box
        confidence = self._conf

        class Box:
            xyxy = [np.array(box, dtype=float)]
            conf = [confidence]

        class Prediction:
            boxes = [Box()]

        return [Prediction()]


def build(box, image_size=(254, 429), **overrides):
    """A recognizer whose detector always returns `box`, recording the crop."""
    settings = Settings(**overrides)
    recognizer = PlateRecognizer(settings)
    recognizer._model = OneBoxModel(box)
    recognizer._ocr_ready = True

    seen: list[np.ndarray] = []

    class Recorder:
        name = 'recorder'

        def load(self):
            pass

        def close(self):
            pass

        def read(self, image):
            seen.append(image)
            return []

    recognizer._ocr = Recorder()
    image = np.full((*image_size, 3), 128, dtype=np.uint8)
    return recognizer, image, seen


def test_padding_scales_with_plate_width():
    """A 320px-wide plate must get far more than the old fixed 10px."""
    recognizer, image, _ = build((59, 49, 379, 150))
    outcome = recognizer.detect(image, annotate=False)

    x1, y1, x2, y2 = outcome.plates[0].box
    pad_left = 59 - x1
    assert pad_left >= 18, f'padding {pad_left}px is too tight to clear a glyph'
    # 6% of the 320px long side.
    assert pad_left == pytest.approx(19, abs=2)
    assert y1 < 49 and x2 > 379 and y2 > 150


def test_small_plate_gets_the_pixel_floor():
    """Proportional padding must not collapse to nothing on a tiny plate."""
    recognizer, image, _ = build((100, 100, 140, 118))
    x1, _, _, _ = recognizer.detect(image, annotate=False).plates[0].box
    assert 100 - x1 == Settings().crop_padding_min


def test_padding_is_clamped_to_the_frame():
    """A plate at the edge must not produce out-of-bounds coordinates."""
    recognizer, image, _ = build((0, 0, 300, 120), image_size=(254, 429))
    x1, y1, x2, y2 = recognizer.detect(image, annotate=False).plates[0].box
    assert x1 == 0 and y1 == 0
    assert x2 <= 429 and y2 <= 254


def test_crop_reaches_the_ocr_engine_with_padding_applied():
    recognizer, image, seen = build((59, 49, 379, 150))
    recognizer.detect(image, annotate=False)
    assert seen, 'OCR was never called'
    height, width = seen[0].shape[:2]
    # 320 + 2*19 = 358 wide, 101 + 2*19 = 139 tall, before any upscaling.
    assert width >= 350
    assert height >= 130


def test_ratio_is_configurable():
    recognizer, image, _ = build((59, 49, 379, 150), crop_padding_ratio=0.12)
    x1, _, _, _ = recognizer.detect(image, annotate=False).plates[0].box
    assert 59 - x1 == pytest.approx(38, abs=2)


def test_zero_area_box_is_skipped():
    recognizer, image, seen = build((10, 10, 10, 10))
    # Padding gives it area, so it should still be attempted rather than crash.
    outcome = recognizer.detect(image, annotate=False)
    assert len(outcome.plates) <= 1
    assert all(p.box[2] >= p.box[0] for p in outcome.plates)
