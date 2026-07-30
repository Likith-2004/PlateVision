"""
Plate parsing, confusion repair and the OCR cascade.

Pure logic, no model. Every case here is derived from real EasyOCR output
observed on the sample images, including the watermark that used to beat the
actual plate on confidence alone.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from app.plates import (
    CASCADE,
    STATE_CODES,
    best_plate,
    build_candidates,
    read_plate_cascade,
    validate_plate,
)


def box(x, y, w=100, h=30):
    """An EasyOCR-style 4-point bounding box."""
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def region(text, conf, x=10, y=10, w=100, h=30):
    """One EasyOCR result tuple."""
    return (box(x, y, w, h), text, conf)


# --------------------------------------------------------------- validation

@pytest.mark.parametrize('text,expected', [
    ('MH12AB1234', 'MH12AB1234'),      # standard, 2-digit RTO
    ('MH 12 AB 1234', 'MH12AB1234'),   # separators stripped
    ('mh12ab1234', 'MH12AB1234'),      # lowercase
    ('DL8CAF5031', 'DL8CAF5031'),      # 1-digit RTO, 3-letter series
    ('KA01MJ2345', 'KA01MJ2345'),
    ('TN07BZ0001', 'TN07BZ0001'),
    ('22BH6517AA', '22BH6517AA'),      # Bharat series
    ('22BH6517', '22BH6517'),          # Bharat, trailing letters missed
])
def test_valid_formats(text, expected):
    result = validate_plate(text)
    assert result['valid'] is True
    assert result['number'] == expected


@pytest.mark.parametrize('text,expected', [
    ('MHI2AB1234', 'MH12AB1234'),      # I -> 1 in a digit slot
    ('MH12A81234', 'MH12AB1234'),      # 8 -> B in a letter slot
    ('MH12AB1O34', 'MH12AB1034'),      # O -> 0
    ('MHI2ABI234', 'MH12AB1234'),      # two repairs
    ('0L8CAF5031', 'DL8CAF5031'),      # 0 -> D, chosen because DL is real
])
def test_confusion_repair(text, expected):
    result = validate_plate(text)
    assert result['valid'] is True
    assert result['number'] == expected
    assert result['corrections'] >= 1


def test_repair_prefers_a_real_state_code():
    """'0' can be D, O or Q; only DL is a state, so DL must win over OL."""
    assert validate_plate('0L8CAF5031')['state'] == 'DL'


@pytest.mark.parametrize('text', [
    'ALAMU',        # dealer watermark seen on a real sample
    'INDIA',
    'IND',
    '',
    'AB',
    '12',
    'HELLOWORLD',   # right length, wrong shape
])
def test_noise_rejected(text):
    result = validate_plate(text)
    assert result['valid'] is False
    assert result['format'] is None


def test_unknown_state_still_parses_but_is_flagged_by_state():
    result = validate_plate('XX99ZZ9999')
    assert result['valid'] is True       # structurally a plate
    assert result['state'] is None       # but XX is not a real state code


def test_none_input():
    assert validate_plate(None)['valid'] is False


def test_state_code_table_is_sane():
    for code in ('MH', 'DL', 'KA', 'TN', 'HR', 'CG', 'UP', 'WB'):
        assert code in STATE_CODES


# --------------------------------------------------------------- candidates

def test_two_line_plate_is_merged():
    """A plate split over two lines must recombine in reading order."""
    ocr = [region('MH 12', 0.80, y=10), region('AB 1234', 0.75, y=60)]
    result = best_plate(ocr)
    assert result['number'] == 'MH12AB1234'
    assert result['valid'] is True


def test_reading_order_is_left_to_right_within_a_row():
    ocr = [region('AB1234', 0.7, x=200, y=10), region('MH12', 0.7, x=10, y=10)]
    assert best_plate(ocr)['number'] == 'MH12AB1234'


def test_candidates_include_singles_and_merges():
    ocr = [region('MH12', 0.8, y=10), region('AB1234', 0.8, y=60)]
    texts = {text for text, _ in build_candidates(ocr)}
    assert 'MH12' in texts
    assert 'AB1234' in texts
    assert 'MH12AB1234' in texts


def test_merge_confidence_is_the_weakest_part():
    ocr = [region('MH12', 0.9, y=10), region('AB1234', 0.4, y=60)]
    merged = dict(build_candidates(ocr))
    assert merged['MH12AB1234'] == pytest.approx(0.4)


# -------------------------------------------------------------- best_plate

def test_valid_plate_beats_higher_confidence_watermark():
    """The regression that motivated all of this: 'alam' at 97% outranked
    the real plate at 43% under plain max-confidence selection."""
    ocr = [region('alam', 0.976, y=10), region('22 BH6517', 0.434, y=60)]
    result = best_plate(ocr)
    assert result['number'] == '22BH6517'
    assert result['valid'] is True


def test_returns_none_when_nothing_usable():
    assert best_plate([]) is None
    assert best_plate([region('xx', 0.9)]) is None


def test_low_confidence_noise_is_dropped():
    assert best_plate([region('QQQQQQQQ', 0.01)]) is None


def test_invalid_text_surfaced_when_confident_enough():
    """Unrecognisable text is returned, but flagged, not silently dropped."""
    result = best_plate([region('IHR26D0555J8', 0.5)])
    assert result is not None
    assert result['valid'] is False


# ----------------------------------------------------------------- cascade

def crop(width=200, height=60):
    return np.full((height, width, 3), 200, dtype=np.uint8)


def test_cascade_stops_on_first_confident_valid_read():
    """A clean plate must cost exactly one OCR call."""
    calls = []

    def ocr(image):
        calls.append(image.shape)
        return [region('MH12AB1234', 0.95)]

    result = read_plate_cascade(crop(), ocr)
    assert result['number'] == 'MH12AB1234'
    assert result['passes'] == 1
    assert result['variant'] == 'clahe'
    assert len(calls) == 1


def test_cascade_escalates_when_first_pass_fails():
    """Pass 1 returns junk; a later variant recovers a real plate."""
    outputs = [
        [region('!!!', 0.2)],                 # clahe: nothing usable
        [region('MH12AB1234', 0.45)],         # clahe+up2x: valid but not confident
        [region('MH12AB1234', 0.48)],         # otsu+up2x: agrees
    ]
    calls = []

    def ocr(image):
        calls.append(image.shape)
        return outputs[min(len(calls) - 1, len(outputs) - 1)]

    result = read_plate_cascade(crop(), ocr, max_passes=3)
    assert result['number'] == 'MH12AB1234'
    assert len(calls) == 3
    assert result['votes'] == 2          # two variants agreed
    assert result['passes'] == 3


def test_cascade_respects_max_passes():
    calls = []

    def ocr(image):
        calls.append(image.shape)
        return [region('!!!', 0.2)]

    read_plate_cascade(crop(), ocr, max_passes=2)
    assert len(calls) == 2


def test_max_passes_one_is_a_single_native_pass():
    """Live/streaming mode must not upscale."""
    shapes = []

    def ocr(image):
        shapes.append(image.shape)
        return [region('MH12AB1234', 0.9)]

    read_plate_cascade(crop(200, 60), ocr, max_passes=1)
    assert len(shapes) == 1
    assert shapes[0][:2] == (60, 200)    # unchanged resolution


def test_later_passes_upscale():
    shapes = []

    def ocr(image):
        shapes.append(image.shape)
        return [region('!!!', 0.2)]

    read_plate_cascade(crop(200, 60), ocr, max_passes=2)
    assert shapes[1][1] > shapes[0][1]   # second pass is wider


def test_upscaling_preserves_aspect_ratio():
    """The old code resized crops to a fixed 400x200, distorting them."""
    ratios = []

    def ocr(image):
        h, w = image.shape[:2]
        ratios.append(round(w / h, 3))
        return [region('!!!', 0.2)]

    read_plate_cascade(crop(300, 100), ocr, max_passes=4)
    assert len(set(ratios)) == 1


def test_voting_prefers_agreement_over_a_lone_high_score():
    outputs = [
        [region('!!!', 0.2)],
        [region('MH12AB1234', 0.30)],
        [region('MH12AB1234', 0.31)],
        [region('KA01MJ2345', 0.60)],   # higher confidence, but alone
    ]
    calls = []

    def ocr(image):
        calls.append(1)
        return outputs[min(len(calls) - 1, len(outputs) - 1)]

    result = read_plate_cascade(crop(), ocr, max_passes=4)
    assert result['number'] == 'MH12AB1234'
    assert result['votes'] == 2


def test_time_budget_stops_escalation():
    """A hopeless crop must not run the whole cascade when the budget is spent."""
    calls = []

    def slow_ocr(image):
        calls.append(1)
        import time
        time.sleep(0.05)
        return [region('!!!', 0.2)]

    read_plate_cascade(crop(), slow_ocr, max_passes=5, time_budget=0.01)
    assert len(calls) == 1      # first pass always runs, then the budget bites


def test_first_pass_always_runs_even_with_zero_budget():
    calls = []

    def ocr(image):
        calls.append(1)
        return [region('MH12AB1234', 0.9)]

    assert read_plate_cascade(crop(), ocr, time_budget=0.0) is not None
    assert len(calls) == 1


def test_ocr_exception_in_one_variant_does_not_abort():
    calls = []

    def flaky(image):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('boom')
        return [region('MH12AB1234', 0.95)]

    result = read_plate_cascade(crop(), flaky, max_passes=2)
    assert result is not None
    assert result['number'] == 'MH12AB1234'


def test_failing_variant_is_logged(caplog):
    """
    A swallowed variant failure silently degrades the answer to a later
    variant's reading, which is exactly how a truncated plate reached the UI
    while every log line looked normal. It must leave a trace.
    """
    calls = []

    def flaky(image):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('kernel compile failed')
        return [region('MH12AB1234', 0.95)]

    with caplog.at_level(logging.WARNING, logger='app.plates'):
        result = read_plate_cascade(crop(), flaky, max_passes=2)

    assert result['passes'] == 2
    assert any('clahe' in r.message for r in caplog.records)
    assert 'kernel compile failed' in caplog.text


def test_ocr_always_failing_returns_none():
    def broken(image):
        raise RuntimeError('boom')

    assert read_plate_cascade(crop(), broken) is None


@pytest.mark.parametrize('bad', [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_cascade_rejects_unusable_crops(bad):
    assert read_plate_cascade(bad, lambda _i: []) is None


def test_grayscale_crop_is_accepted():
    """Callers may hand in a single-channel image."""
    gray = np.full((60, 200), 128, dtype=np.uint8)
    result = read_plate_cascade(gray, lambda _i: [region('MH12AB1234', 0.9)])
    assert result['number'] == 'MH12AB1234'


def test_cascade_order_is_cheapest_first():
    """Pass 1 must be the native-resolution variant; upscaling is a fallback."""
    assert CASCADE[0][0] == 'clahe'
    assert 'up' in CASCADE[1][0]
