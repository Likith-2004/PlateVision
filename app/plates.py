"""
Indian number plate post-processing for raw EasyOCR output.

EasyOCR returns generic text; on a plate crop that means watermarks, partial
reads, split lines, and the usual O/0 I/1 S/5 confusions all arrive mixed
together with no indication of which is the plate. This module resolves them
into a single best answer using the structure of Indian plate formats.

Pipeline:
  1. Assemble candidate strings from OCR regions in reading order, so a
     two-line plate ("MH12" / "AB1234") is also tried as one string.
  2. Coerce each candidate onto every plausible plate template, fixing
     letter/digit confusions only where the template demands it.
  3. Score by format validity first, OCR confidence second, and penalise
     each character that had to be corrected.

Deliberately free of FastAPI, settings and the OCR engine: it takes raw OCR
tuples in and returns plain dicts out, which keeps it directly unit-testable
without loading a 52MB model. See tests/test_plates.py.
"""

from __future__ import annotations

import itertools
import logging
import re
import time
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

# State / UT codes, including retired ones still on the road (OR, UA, TG).
STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'TG',
    'UA', 'UK', 'UP', 'UT', 'WB',
}

# Standard 1988 series: <state 2A> <rto 1-2D> <series 0-3A> <number 4D>
#   e.g. MH12AB1234, DL8CAF5031, KA01MJ2345
RE_STANDARD = re.compile(r'^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$')

# Bharat (BH) series, 2021+: <year 2D> BH <number 4D> <series 1-2A>
#   e.g. 22BH6517A. Trailing letters are small and often missed, so allow 0.
RE_BH = re.compile(r'^(\d{2})(BH)(\d{4})([A-Z]{0,2})$')

# Minimum plausible plate length after cleaning; below this it is noise.
MIN_PLATE_LEN = 8

# Ceiling on the share of characters that may be rewritten by confusion repair,
# so coercion cannot invent a plate out of unrelated text.
MAX_CORRECTION_RATIO = 0.3

# Character confusions, applied only where a template requires the other class.
# Each digit lists every letter it can be mistaken for; scoring then picks the
# reading that yields a real state code (so "0L8CAF5031" resolves to DL, not OL).
TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'J': '1',
            'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
TO_ALPHA = {'0': 'ODQ', '1': 'IL', '2': 'Z', '3': 'J', '4': 'A', '5': 'S',
            '6': 'G', '7': 'T', '8': 'B'}

# Cap on alternative readings per template; well above the real maximum
# (3 options over at most 5 letter slots) but bounds pathological input.
_MAX_VARIANTS = 512


def _templates(length):
    """Plate templates of exactly `length` chars. 'A' = letter, 'D' = digit."""
    out = []
    # Standard: prefer 2-digit RTO and 2-letter series (by far most common).
    for n_digits in (2, 1):
        for n_series in (2, 3, 1, 0):
            t = 'AA' + 'D' * n_digits + 'A' * n_series + 'DDDD'
            if len(t) == length:
                out.append(('standard', t))
    # BH series.
    for n_series in (2, 1, 0):
        t = 'DDAA' + 'DDDD' + 'A' * n_series
        if len(t) == length:
            out.append(('bh_series', t))
    return out


def _coerce(text, template):
    """
    Force `text` onto `template`.

    Yields every (result, n_corrections) reading reachable via the confusion
    maps, since an ambiguous digit like '0' may be D, O or Q and only the
    state-code check can tell which. Empty if the text cannot fit at all.
    """
    per_position = []
    for ch, kind in zip(text, template):
        if kind == 'D':
            if ch.isdigit():
                opts = [(ch, 0)]
            elif ch in TO_DIGIT:
                opts = [(TO_DIGIT[ch], 1)]
            else:
                return []
        else:  # 'A'
            if ch.isalpha():
                opts = [(ch, 0)]
            elif ch in TO_ALPHA:
                opts = [(alt, 1) for alt in TO_ALPHA[ch]]
            else:
                return []
        per_position.append(opts)

    readings = []
    for combo in itertools.product(*per_position):
        readings.append((''.join(c for c, _ in combo),
                         sum(f for _, f in combo)))
        if len(readings) >= _MAX_VARIANTS:
            break
    return readings


def validate_plate(text: str | None) -> dict:
    """
    Interpret `text` as an Indian plate.

    Returns a dict: number, valid, format, corrections, state.
    `number` is the corrected plate when valid, else the cleaned input.
    """
    cleaned = re.sub(r'[^A-Z0-9]', '', (text or '').upper())
    fallback = {'number': cleaned, 'valid': False, 'format': None,
                'corrections': 0, 'state': None}

    if len(cleaned) < MIN_PLATE_LEN:
        return fallback

    # Cap how much of the string may be rewritten. Given enough substitutions
    # almost any text can be forced into a plate shape -- "QQQQQQQQ" becomes
    # "QQ0Q0000" with five edits -- and that is fabrication, not repair. Real
    # confusions affect a character or two.
    max_fixes = max(2, int(len(cleaned) * MAX_CORRECTION_RATIO))

    best = None
    for fmt, template in _templates(len(cleaned)):
        for candidate, fixes in _coerce(cleaned, template):
            if fixes > max_fixes:
                continue
            if fmt == 'standard':
                m = RE_STANDARD.match(candidate)
                if not m:
                    continue
                code = m.group(1)
                # An unrecognised state code usually means this is not a plate.
                known = code in STATE_CODES
                # Only report a state when it is a real one; echoing "XX" back
                # would imply a lookup succeeded when it did not.
                state = code if known else None
                score = (100 if known else 55) - fixes * 4
            else:
                m = RE_BH.match(candidate)
                if not m:
                    continue
                state = None
                score = 95 - fixes * 4

            if best is None or score > best[0]:
                best = (score, {'number': candidate, 'valid': True,
                                'format': fmt, 'corrections': fixes,
                                'state': state})

    return best[1] if best else fallback


# ---------------------------------------------------------------------------
# Candidate assembly from EasyOCR regions
# ---------------------------------------------------------------------------

def _region_geometry(box):
    """(y_center, x_left) for an EasyOCR 4-point box."""
    ys = [p[1] for p in box]
    xs = [p[0] for p in box]
    return sum(ys) / len(ys), min(xs)


def _reading_order(ocr_results):
    """Sort regions top-to-bottom then left-to-right, grouping into rows."""
    items = []
    for entry in ocr_results:
        if len(entry) < 3:
            continue
        box, text, conf = entry[0], str(entry[1]), float(entry[2])
        y, x = _region_geometry(box)
        heights = [p[1] for p in box]
        items.append({'text': text, 'conf': conf, 'y': y, 'x': x,
                      'h': max(heights) - min(heights) or 1})
    if not items:
        return []

    items.sort(key=lambda i: i['y'])
    row_tol = max(i['h'] for i in items) * 0.6
    rows, current = [], [items[0]]
    for it in items[1:]:
        if abs(it['y'] - current[-1]['y']) <= row_tol:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    rows.append(current)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda i: i['x']))
    return ordered


def build_candidates(ocr_results: Iterable[Sequence]) -> list[tuple[str, float]]:
    """
    Build (text, confidence) candidates from EasyOCR output.

    Includes each region alone, the full concatenation, and adjacent runs, so
    plates split across two lines or separated by the state emblem recombine.
    """
    ordered = _reading_order(ocr_results)
    if not ordered:
        return []

    candidates = []
    seen = set()

    def add(text, conf):
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append((cleaned, conf))

    # Individual regions.
    for it in ordered:
        add(it['text'], it['conf'])

    # Contiguous runs of 2+ regions (a split plate is the usual cause).
    n = len(ordered)
    for start in range(n):
        for end in range(start + 2, n + 1):
            run = ordered[start:end]
            joined = ''.join(r['text'] for r in run)
            # Confidence of a merge is only as good as its weakest part.
            add(joined, min(r['conf'] for r in run))

    return candidates


def best_plate(ocr_results: Iterable[Sequence],
               min_confidence: float = 0.1) -> dict | None:
    """
    Pick the single best plate reading from raw EasyOCR output.

    Returns None if nothing usable was found, else a dict with:
      number, confidence (%), valid, format, corrections, state
    Format-valid readings always beat higher-confidence non-plate text, which
    is what stops watermarks and slogans from winning.
    """
    scored = []
    for text, conf in build_candidates(ocr_results):
        info = validate_plate(text)

        if info['valid']:
            # Valid format dominates; confidence and edit count break ties.
            rank = 1_000_000 + conf * 1000 - info['corrections'] * 50
        else:
            if conf < min_confidence or len(info['number']) < MIN_PLATE_LEN:
                continue
            rank = conf * 1000

        scored.append((rank, {
            'number': info['number'],
            'confidence': round(conf * 100, 2),
            'valid': info['valid'],
            'format': info['format'],
            'corrections': info['corrections'],
            'state': info['state'],
        }))

    if not scored:
        return None
    return max(scored, key=lambda s: s[0])[1]


# ---------------------------------------------------------------------------
# Preprocessing cascade
# ---------------------------------------------------------------------------
#
# EasyOCR is sensitive to glyph size and contrast, and no single preprocessing
# wins on every plate. Measured over the sample crops:
#
#   variant           OIP.jpeg    OIP_2.webp        car-plate
#   clahe (old)       81.7%       22.7%  invalid    44.9%  invalid
#   clahe+up2x        96.2%       25.5%  VALID      24.6%  invalid
#   clahe+up3x        97.1%       31.3%  VALID      26.7%  invalid
#   otsu+up2x         76.7%       28.0%  VALID      50.5%  invalid
#
# Upscaling is the big lever, but it also costs the most: 2x the width is 4x
# the pixels, and OCR time tracks area. Since a correct *number* matters and
# the confidence figure is cosmetic, the cascade is ordered cheapest-first and
# stops as soon as a pass yields a confident format-valid plate. Clean plates
# therefore cost exactly one native-resolution pass, the same as before this
# cascade existed; only crops that actually fail pay for upscaling.

def _clahe(gray_img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_img)


def _gray(crop):
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop


def _upscale(img, factor, max_width=1600):
    """Enlarge by `factor`, preserving aspect ratio. Never distorts."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    if w * factor > max_width:
        factor = max(1.0, max_width / w)
        if factor <= 1:
            return img
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def _v_clahe(crop):
    return _clahe(_gray(crop))


def _v_clahe_up2(crop):
    return _upscale(_clahe(_gray(crop)), 2)


def _v_otsu_up2(crop):
    img = _upscale(_clahe(_gray(crop)), 2)
    _, thresholded = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def _v_clahe_up3(crop):
    return _upscale(_clahe(_gray(crop)), 3)


def _v_gray(crop):
    # No CLAHE: contrast enhancement amplifies background text such as dealer
    # watermarks, so a plain grayscale pass is a useful last opinion.
    return _gray(crop)


# Ordered cheapest-first. Every entry after the first roughly quadruples the
# pixel count, so they only run when earlier passes fail to produce a plate.
CASCADE = (
    ('clahe', _v_clahe),
    ('clahe+up2x', _v_clahe_up2),
    ('otsu+up2x', _v_otsu_up2),
    ('clahe+up3x', _v_clahe_up3),
    ('gray', _v_gray),
)

# Passes attempted by default. Bounded so a hopeless crop cannot stall a
# request: the remaining variants stay available via max_passes.
DEFAULT_MAX_PASSES = 3


def read_plate_cascade(crop: np.ndarray | None,
                       ocr_fn: Callable[[np.ndarray], Iterable[Sequence]],
                       min_confidence: float = 0.1,
                       accept_confidence: float = 0.70,
                       max_passes: int | None = DEFAULT_MAX_PASSES,
                       time_budget: float | None = None) -> dict | None:
    """
    Read a plate from `crop`, escalating through preprocessing variants.

    `ocr_fn(image)` must return raw EasyOCR results; the caller supplies it so
    this module stays free of the reader instance and any threading lock.

    Stops early once a pass returns a format-valid reading at or above
    `accept_confidence`, so clean plates cost a single native-resolution OCR
    call. If no pass clears that bar, every format-valid reading collected is
    pooled and voted on, weighting agreement between variants over any single
    confidence score.

    Returns the same shape as `best_plate`, plus:
      variant  - which preprocessing produced the answer
      passes   - how many OCR calls were spent
      votes    - how many variants independently agreed on this number
    Returns None when nothing usable was found.
    """
    if crop is None or getattr(crop, 'size', 0) == 0:
        return None

    stages = CASCADE if max_passes is None else CASCADE[:max(1, max_passes)]
    collected = []
    started = time.monotonic()
    spent = 0

    for index, (name, transform) in enumerate(stages, start=1):
        # The first pass always runs; later ones only while the budget holds,
        # so an unreadable crop cannot stretch a request indefinitely.
        if time_budget and index > 1 and time.monotonic() - started >= time_budget:
            break
        spent = index
        try:
            candidate = best_plate(ocr_fn(transform(crop)),
                                   min_confidence=min_confidence)
        except Exception:
            # A single bad variant must not sink the whole read, but it must not
            # vanish either: silently skipping pass 1 quietly degrades every
            # result to a later variant's answer.
            logger.warning('OCR variant %r failed; continuing', name, exc_info=True)
            continue
        if candidate is None:
            continue

        candidate = dict(candidate, variant=name, passes=index, votes=1)
        collected.append(candidate)

        if candidate['valid'] and candidate['confidence'] >= accept_confidence * 100:
            return candidate

    if not collected:
        return None

    valid = [c for c in collected if c['valid']]
    if not valid:
        # Nothing matched a plate format; surface the most confident attempt so
        # the caller can still show it, flagged invalid.
        best = max(collected, key=lambda c: c['confidence'])
        best['passes'] = spent
        return best

    # Vote: agreement across independent variants outweighs a lone high score.
    groups = {}
    for c in valid:
        groups.setdefault(c['number'], []).append(c)

    def group_rank(members):
        agreement = len(members)
        confidence = max(m['confidence'] for m in members)
        fixes = min(m['corrections'] for m in members)
        return (agreement * 60) + confidence - (fixes * 5)

    winner_number = max(groups, key=lambda n: group_rank(groups[n]))
    members = groups[winner_number]
    winner = max(members, key=lambda m: m['confidence'])
    winner['votes'] = len(members)
    winner['passes'] = spent
    return winner
