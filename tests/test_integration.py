"""
End-to-end tests against the real YOLO weights and EasyOCR.

Marked `slow` and skipped automatically when the weights are absent, so the
fast suite still runs on a clean checkout. Run explicitly with:

    pytest -m slow
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.inference import ImageDecodeError, PlateRecognizer, decode_image
from app.main import create_app
from tests.conftest import SAMPLES

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def recognizer(request):
    from app.config import Settings

    settings = Settings()
    if not settings.model_path.exists():
        pytest.skip(f'model weights not present at {settings.model_path}')

    engine = PlateRecognizer(settings)
    engine.load()
    request.addfinalizer(engine.close)
    return engine


def _read(name: str) -> bytes:
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f'sample {name} not present')
    return path.read_bytes()


def test_reads_the_bharat_series_plate(recognizer):
    """
    A clean plate must be read correctly, and cost a single OCR pass.

    The expected value includes the trailing series letter. EasyOCR dropped it
    and read only "22BH6517"; PaddleOCR reads the full "22BH6517A" at ~99%,
    which matches the Bharat-series format ##BH####XX.
    """
    outcome = recognizer.detect(decode_image(_read('plate_bh_series.jpeg')))

    assert len(outcome.plates) == 1
    reading = outcome.plates[0].reading
    assert reading is not None
    assert reading['number'] == '22BH6517A'
    assert reading['valid'] is True
    assert reading['format'] == 'bh_series'
    # Cheapest variant resolves it, so no upscaling is paid for.
    assert reading['passes'] == 1
    assert reading['variant'] == 'clahe'


def test_watermark_does_not_win(recognizer):
    """The sample carries dealer text that OCR reads more confidently than the
    plate; format validation must still pick the plate."""
    outcome = recognizer.detect(decode_image(_read('plate_bh_series.jpeg')))
    numbers = [p.reading['number'] for p in outcome.plates if p.reading]
    assert 'ALAMU' not in numbers
    assert 'ALAM' not in numbers


def test_hard_plate_is_still_resolved(recognizer):
    """
    A low-resolution plate must still reach a format-valid reading.

    How it gets there is engine-dependent and deliberately not asserted:
    EasyOCR needs the cascade to escalate to pass 3, while PaddleOCR resolves
    it on the first native pass. The escalation mechanism itself is covered by
    unit tests in test_plates.py.
    """
    outcome = recognizer.detect(decode_image(_read('plate_hr_lowres.webp')))

    assert outcome.plates
    reading = outcome.plates[0].reading
    assert reading is not None
    assert reading['valid'] is True
    assert reading['number'].startswith('HR26')


def test_crop_padding_does_not_truncate_the_last_character(recognizer):
    """
    Regression: a fixed 10px crop margin clipped the final glyph.

    This plate read as "CG04H880" -- eight characters, matching no format -- and
    was wrongly diagnosed as a damaged plate, because two OCR engines
    independently agreed on the *clipped* crop. With padding proportional to the
    box, the fourth digit is inside the crop and it resolves as a valid
    Chhattisgarh plate. Detector boxes hug the glyphs, so the margin has to
    scale with them.
    """
    outcome = recognizer.detect(decode_image(_read('plate_partial.jpg')))

    assert outcome.plates
    reading = outcome.plates[0].reading
    assert reading is not None
    assert reading['number'] == 'CG04H8801'
    assert reading['valid'] is True
    assert reading['format'] == 'standard'
    assert reading['state'] == 'CG'


def test_detection_confidence_is_reported(recognizer):
    outcome = recognizer.detect(decode_image(_read('plate_bh_series.jpeg')))
    assert 50 < outcome.plates[0].detection_confidence <= 100


def test_annotation_does_not_mutate_the_input(recognizer):
    frame = decode_image(_read('plate_bh_series.jpeg'))
    before = frame.copy()
    recognizer.detect(frame, annotate=True)
    assert (frame == before).all()


def test_large_image_is_downscaled_without_breaking_boxes(recognizer):
    """Regression: a bad reshape used to collapse the frame width to 3,
    clamping every bounding box to nothing."""
    import cv2

    frame = decode_image(_read('plate_bh_series.jpeg'))
    big = cv2.resize(frame, (1920, 1137))

    outcome = recognizer.detect(big)
    assert outcome.width > 100 and outcome.height > 100
    assert outcome.plates
    x1, y1, x2, y2 = outcome.plates[0].box
    assert x2 - x1 > 50
    assert y2 - y1 > 20


def test_both_backends_read_the_clean_plate(recognizer):
    """
    Real-engine parity check for the pluggable layer.

    Both backends must produce a format-valid Bharat-series reading of the same
    plate. They are allowed to differ in completeness -- EasyOCR drops the
    trailing series letter -- so the assertion is on the shared prefix.
    """
    from app.config import Settings
    from app.inference import PlateRecognizer

    payload = _read('plate_bh_series.jpeg')

    default = recognizer.detect(decode_image(payload)).plates[0].reading
    assert default['valid'] is True
    assert default['format'] == 'bh_series'

    other_engine = 'easyocr' if recognizer.ocr_engine == 'paddleocr' else 'paddleocr'
    alternate = PlateRecognizer(Settings(ocr_engine=other_engine))
    alternate.load()
    try:
        assert alternate.ocr_engine == other_engine
        second = alternate.detect(decode_image(payload)).plates[0].reading
    finally:
        alternate.close()

    assert second['valid'] is True
    assert second['format'] == 'bh_series'
    assert second['number'].startswith('22BH6517')
    assert default['number'].startswith('22BH6517')


def test_decode_rejects_junk():
    with pytest.raises(ImageDecodeError):
        decode_image(b'')
    with pytest.raises(ImageDecodeError):
        decode_image(b'definitely not an image')


def test_full_http_round_trip(recognizer):
    """The real pipeline through the actual API surface."""
    from app.config import Settings

    app = create_app(Settings())
    with TestClient(app) as client:
        # Models load in a background task, so the server is reachable before
        # it is ready. Poll rather than assuming.
        for _ in range(120):
            if client.get('/health').json()['ready']:
                break
            time.sleep(0.5)
        else:
            pytest.fail('models did not become ready in 60s')

        response = client.post(
            '/detect',
            files={'image': ('plate.jpeg', _read('plate_bh_series.jpeg'), 'image/jpeg')},
        )
        assert response.status_code == 200
        body = response.json()
        assert body['plates'][0]['number'] == '22BH6517A'
        assert body['plates'][0]['valid'] is True

        image = client.get(body['image_url'])
        assert image.status_code == 200
        assert image.content[:2] == b'\xff\xd8'
