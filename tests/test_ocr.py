"""
OCR backend selection and output normalisation.

The adapters are tested against fakes rather than the real engines: what matters
downstream is that both produce identical tuple shapes, and that is exactly
where a subtle mismatch would hide.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.ocr import (
    BACKENDS,
    EasyOCRBackend,
    PaddleOCRBackend,
    _as_bgr,
    create_backend,
)


# ------------------------------------------------------------------ selection

def test_paddle_is_the_default():
    assert Settings().ocr_engine == 'paddleocr'
    assert create_backend(Settings()).name == 'paddleocr'


def test_easyocr_is_selectable():
    backend = create_backend(Settings(ocr_engine='easyocr'))
    assert backend.name == 'easyocr'


def test_unknown_engine_is_rejected_at_startup():
    """A typo must fail loudly at config parse time, not at first request."""
    with pytest.raises(Exception):
        Settings(ocr_engine='tesseract')


def test_create_backend_rejects_unknown_engine():
    class Fake:
        ocr_engine = 'nope'
        ocr_languages = ['en']
        ocr_gpu = False
        paddle_detection_model = 'd'
        paddle_recognition_model = 'r'

    with pytest.raises(ValueError, match='unknown ocr_engine'):
        create_backend(Fake())


def test_paddle_mobile_models_are_pinned():
    """The 3.x server defaults are ~40x slower on CPU; never inherit them."""
    settings = Settings()
    assert 'mobile' in settings.paddle_detection_model
    assert 'mobile' in settings.paddle_recognition_model


def test_backends_are_declared():
    assert set(BACKENDS) == {'paddleocr', 'easyocr'}


# -------------------------------------------------------------- colour handling

def test_grayscale_is_promoted_to_three_channels():
    """The cascade emits grayscale and thresholded crops; paddle needs colour."""
    gray = np.zeros((10, 20), dtype=np.uint8)
    assert _as_bgr(gray).shape == (10, 20, 3)


def test_colour_images_pass_through_untouched():
    colour = np.zeros((10, 20, 3), dtype=np.uint8)
    assert _as_bgr(colour) is colour


# ------------------------------------------------------------ easyocr adapter

class FakeEasyReader:
    def __init__(self, results):
        self._results = results
        self.seen = []

    def readtext(self, image):
        self.seen.append(image.shape)
        return self._results


def test_easyocr_output_is_normalised():
    """numpy scalars must become plain Python floats."""
    backend = EasyOCRBackend()
    backend._reader = FakeEasyReader([
        ([[0, 0], [10, 0], [10, 5], [0, 5]], 'MH12AB1234', np.float64(0.87)),
    ])

    (box, text, conf), = backend.read(np.zeros((5, 10), dtype=np.uint8))
    assert text == 'MH12AB1234'
    assert isinstance(conf, float) and not isinstance(conf, np.floating)
    assert conf == pytest.approx(0.87)
    assert all(isinstance(v, float) for point in box for v in point)


def test_easyocr_skips_malformed_entries():
    backend = EasyOCRBackend()
    backend._reader = FakeEasyReader([
        ([[0, 0]], 'too-short'),                                  # no confidence
        ([[0, 0], [1, 0], [1, 1], [0, 1]], 'MH12AB1234', 0.9),
    ])
    assert len(backend.read(np.zeros((5, 10), dtype=np.uint8))) == 1


def test_unloaded_easyocr_backend_raises():
    with pytest.raises(RuntimeError, match='not loaded'):
        EasyOCRBackend().read(np.zeros((5, 10), dtype=np.uint8))


# ------------------------------------------------------------- paddle adapter

class FakePaddle:
    def __init__(self, results):
        self._results = results
        self.seen = []

    def predict(self, input):           # noqa: A002 - mirrors paddle's kwarg
        self.seen.append(input.shape)
        return self._results


def test_paddle_output_is_normalised():
    backend = PaddleOCRBackend()
    backend._ocr = FakePaddle([{
        'rec_texts': ['22BH6517A', 'IND'],
        'rec_scores': [0.991, 0.994],
        'rec_polys': [
            [[5, 5], [95, 5], [95, 30], [5, 30]],
            [[5, 35], [40, 35], [40, 55], [5, 55]],
        ],
    }])

    readings = backend.read(np.zeros((60, 100), dtype=np.uint8))
    assert [text for _, text, _ in readings] == ['22BH6517A', 'IND']
    assert readings[0][2] == pytest.approx(0.991)
    assert readings[0][0][0] == [5.0, 5.0]


def test_paddle_falls_back_to_dt_polys():
    """3.x exposes rec_polys, but dt_polys is the raw-detection equivalent."""
    backend = PaddleOCRBackend()
    backend._ocr = FakePaddle([{
        'rec_texts': ['MH12AB1234'],
        'rec_scores': [0.9],
        'dt_polys': [[[0, 0], [10, 0], [10, 5], [0, 5]]],
    }])
    (box, text, _), = backend.read(np.zeros((5, 10), dtype=np.uint8))
    assert text == 'MH12AB1234'
    assert box[1] == [10.0, 0.0]


def test_paddle_tolerates_missing_polygons():
    backend = PaddleOCRBackend()
    backend._ocr = FakePaddle([{'rec_texts': ['MH12AB1234'], 'rec_scores': [0.9]}])
    (box, text, _), = backend.read(np.zeros((5, 10), dtype=np.uint8))
    assert text == 'MH12AB1234'
    assert len(box) == 4


def test_paddle_tolerates_missing_scores():
    backend = PaddleOCRBackend()
    backend._ocr = FakePaddle([{'rec_texts': ['MH12AB1234']}])
    (_, _, conf), = backend.read(np.zeros((5, 10), dtype=np.uint8))
    assert conf == 0.0


def test_paddle_handles_empty_result():
    backend = PaddleOCRBackend()
    backend._ocr = FakePaddle([{'rec_texts': [], 'rec_scores': []}])
    assert backend.read(np.zeros((5, 10), dtype=np.uint8)) == []


def test_paddle_receives_three_channel_input():
    backend = PaddleOCRBackend()
    fake = FakePaddle([{'rec_texts': [], 'rec_scores': []}])
    backend._ocr = fake
    backend.read(np.zeros((5, 10), dtype=np.uint8))
    assert fake.seen == [(5, 10, 3)]


def test_unloaded_paddle_backend_raises():
    with pytest.raises(RuntimeError, match='not loaded'):
        PaddleOCRBackend().read(np.zeros((5, 10), dtype=np.uint8))


def test_pinned_models_omit_lang(monkeypatch):
    """Paddle ignores `lang` once model names are set, and warns; don't send it."""
    captured = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import paddleocr
    monkeypatch.setattr(paddleocr, 'PaddleOCR', FakePaddleOCR)

    PaddleOCRBackend(language='en',
                     detection_model='PP-OCRv5_mobile_det',
                     recognition_model='PP-OCRv5_mobile_rec').load()

    assert 'lang' not in captured
    assert captured['text_detection_model_name'] == 'PP-OCRv5_mobile_det'
    assert captured['use_doc_unwarping'] is False


def test_blank_models_fall_back_to_lang(monkeypatch):
    """Clearing the model names selects by language instead."""
    captured = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import paddleocr
    monkeypatch.setattr(paddleocr, 'PaddleOCR', FakePaddleOCR)

    PaddleOCRBackend(language='en', detection_model='', recognition_model='').load()

    assert captured['lang'] == 'en'
    assert 'text_detection_model_name' not in captured


# ------------------------------------------------- interchangeability contract

def test_both_backends_feed_the_resolver_identically():
    """The same plate, via either adapter, must resolve to the same reading."""
    from app.plates import best_plate

    box = [[5, 5], [95, 5], [95, 30], [5, 30]]

    easy = EasyOCRBackend()
    easy._reader = FakeEasyReader([(box, '22 BH6517A', np.float64(0.82))])

    paddle = PaddleOCRBackend()
    paddle._ocr = FakePaddle([{
        'rec_texts': ['22BH6517A'], 'rec_scores': [0.99], 'rec_polys': [box],
    }])

    image = np.zeros((40, 100), dtype=np.uint8)
    from_easy = best_plate(easy.read(image))
    from_paddle = best_plate(paddle.read(image))

    assert from_easy['number'] == from_paddle['number'] == '22BH6517A'
    assert from_easy['valid'] and from_paddle['valid']
    assert from_easy['format'] == from_paddle['format'] == 'bh_series'
