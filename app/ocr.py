"""
Pluggable OCR backends.

Both engines are normalised to EasyOCR's tuple shape:

    [(box_4points, text, confidence), ...]

which is what `app.plates` consumes. Nothing downstream knows or cares which
engine produced a reading, so switching is a config change.

Measured on the sample plates (single pass, same YOLO crops):

    engine                    load     per plate   readings
    easyocr                   12.2s    1.4-3.1s    1 of 3 format-valid
    paddleocr (server tier)  208.1s     13-15s     3 of 3, ~97%
    paddleocr (mobile tier)   19.6s    0.29-0.41s  3 of 3, ~99%

Hence the default. Note the middle row: PaddleOCR 3.x selects server-grade
models unless told otherwise, which is ~40x slower here for no accuracy gain,
so the mobile model names are pinned explicitly below.

Engine imports are deferred into `load()` because both are expensive
(paddleocr costs ~20s of import alone) and only one is ever used.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Reading = tuple[list[list[float]], str, float]

_FALLBACK_BOX = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Promote grayscale to 3-channel.

    The preprocessing cascade hands over grayscale and Otsu-thresholded
    single-channel images; PaddleOCR expects colour.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class OCRBackend(ABC):
    """One text-recognition engine."""

    name: str = 'unknown'

    @abstractmethod
    def load(self) -> None:
        """Import and initialise the engine. Blocking, called once."""

    @abstractmethod
    def read(self, image: np.ndarray) -> list[Reading]:
        """Recognise text, normalised to (box, text, confidence) tuples."""

    def close(self) -> None:
        pass


class EasyOCRBackend(OCRBackend):
    name = 'easyocr'

    def __init__(self, languages: Sequence[str] = ('en',), gpu: bool = False) -> None:
        self._languages = list(languages)
        self._gpu = gpu
        self._reader = None

    def load(self) -> None:
        import easyocr

        started = time.monotonic()
        self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        logger.info('easyocr ready (%s, gpu=%s) in %.1fs',
                    ','.join(self._languages), self._gpu,
                    time.monotonic() - started)

    def read(self, image: np.ndarray) -> list[Reading]:
        if self._reader is None:
            raise RuntimeError('backend not loaded')
        # EasyOCR already returns (box, text, confidence); normalise the numeric
        # types, since it hands back numpy floats.
        out: list[Reading] = []
        for entry in self._reader.readtext(image):
            if len(entry) < 3:
                continue
            box, text, confidence = entry[0], entry[1], entry[2]
            out.append((
                [[float(x), float(y)] for x, y in box],
                str(text),
                float(confidence),
            ))
        return out

    def close(self) -> None:
        self._reader = None


def _prepare_cache(seed: Path | None, cache: Path | None) -> Path | None:
    """
    Point PaddleOCR at a writable cache, seeded from a bundled copy.

    PaddleOCR downloads its weights on first construction into
    `$PADDLE_PDX_CACHE_HOME` (default `~/.paddlex`). Neither half of that
    default survives a serverless host: `$HOME` is read-only, and re-downloading
    on every cold start would add a network hop to the slowest path there is.

    So `scripts/warm_paddle.py` bakes the models into `seed` at build time, and
    this copies them into `cache` (somewhere writable, e.g. /tmp) once per
    process. ~21 MB of local copy replaces a download.

    Returns the directory actually used, or None to leave Paddle's own default
    alone -- which is the right behaviour for a normal local run.
    """
    if cache is None:
        return None

    try:
        cache.mkdir(parents=True, exist_ok=True)
        if seed is not None and seed.is_dir():
            models = seed / 'official_models'
            destination = cache / 'official_models'
            if models.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                copied = []
                for source in models.iterdir():
                    if not source.is_dir() or (destination / source.name).is_dir():
                        continue
                    shutil.copytree(source, destination / source.name)
                    copied.append(source.name)
                if copied:
                    logger.info('seeded paddle cache from %s: %s',
                                models, ', '.join(copied))
    except OSError as exc:
        # Fall back to letting Paddle download. Slower, but a permissions
        # problem here should not take the whole service down.
        logger.warning('could not prepare paddle cache at %s: %s', cache, exc)
        return None

    # Read at import time by paddlex.utils.cache, so it must be set before the
    # `from paddleocr import PaddleOCR` below.
    os.environ['PADDLE_PDX_CACHE_HOME'] = str(cache)
    return cache


class PaddleOCRBackend(OCRBackend):
    name = 'paddleocr'

    def __init__(self, language: str = 'en',
                 detection_model: str = 'PP-OCRv5_mobile_det',
                 recognition_model: str = 'PP-OCRv5_mobile_rec',
                 model_dir: Path | None = None,
                 cache_dir: Path | None = None) -> None:
        self._language = language
        self._detection_model = detection_model
        self._recognition_model = recognition_model
        self._model_dir = model_dir
        self._cache_dir = cache_dir
        self._ocr = None

    def load(self) -> None:
        cache = _prepare_cache(self._model_dir, self._cache_dir)

        from paddleocr import PaddleOCR

        options: dict = {
            # Whole-page document features; meaningless on a cropped plate and
            # each one costs an extra model load plus inference.
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }

        if self._detection_model and self._recognition_model:
            # Pinned explicitly: the 3.x defaults are server-grade and ~40x
            # slower on CPU with no accuracy benefit on plate crops. Paddle
            # ignores `lang` once model names are given -- and warns about it --
            # so it is deliberately omitted here; the model name carries the
            # language. Blank both names to select by language instead.
            options['text_detection_model_name'] = self._detection_model
            options['text_recognition_model_name'] = self._recognition_model
            selected = f'det={self._detection_model}, rec={self._recognition_model}'
        else:
            options['lang'] = self._language
            selected = f'lang={self._language} (default model tier)'

        started = time.monotonic()
        self._ocr = PaddleOCR(**options)
        logger.info('paddleocr ready (%s, cache=%s) in %.1fs',
                    selected, cache or 'default',
                    time.monotonic() - started)

    def read(self, image: np.ndarray) -> list[Reading]:
        if self._ocr is None:
            raise RuntimeError('backend not loaded')

        out: list[Reading] = []
        for result in self._ocr.predict(input=_as_bgr(image)):
            texts = result.get('rec_texts') or []
            scores = result.get('rec_scores') or []
            # 3.x exposes rectified polygons as rec_polys and raw detections as
            # dt_polys; either is fine for reading order.
            polys = result.get('rec_polys')
            if polys is None:
                polys = result.get('dt_polys') or []

            for index, text in enumerate(texts):
                confidence = float(scores[index]) if index < len(scores) else 0.0
                if index < len(polys):
                    box = [[float(x), float(y)] for x, y in polys[index]]
                else:
                    box = [row[:] for row in _FALLBACK_BOX]
                out.append((box, str(text), confidence))
        return out

    def close(self) -> None:
        self._ocr = None


BACKENDS = ('paddleocr', 'easyocr')


def create_backend(settings) -> OCRBackend:
    """Build the backend named by `settings.ocr_engine`."""
    engine = settings.ocr_engine
    if engine == 'easyocr':
        return EasyOCRBackend(languages=settings.ocr_languages,
                              gpu=settings.ocr_gpu)
    if engine == 'paddleocr':
        language = settings.ocr_languages[0] if settings.ocr_languages else 'en'
        return PaddleOCRBackend(
            language=language,
            detection_model=settings.paddle_detection_model,
            recognition_model=settings.paddle_recognition_model,
            model_dir=settings.paddle_model_dir,
            cache_dir=settings.paddle_cache_dir,
        )
    raise ValueError(f'unknown ocr_engine {engine!r}; expected one of {BACKENDS}')
