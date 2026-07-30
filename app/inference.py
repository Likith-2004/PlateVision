"""
Detection + OCR engine.

Holds the YOLO detector and an OCR backend (see app/ocr.py), loads each exactly
once, and exposes a single synchronous `detect()` entry point.

Threading note: neither ultralytics nor the OCR engines are thread-safe, and CPU
inference gains nothing from oversubscription, so all model access is
serialised behind one lock. The async layer keeps the event loop free by
running `detect()` in a worker thread (see routes.py), which means a slow
request never blocks unrelated ones from being accepted and queued.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.ocr import OCRBackend, create_backend
from app.plates import read_plate_cascade

logger = logging.getLogger(__name__)


class ImageDecodeError(ValueError):
    """Raised when bytes cannot be decoded into an image."""


@dataclass(slots=True)
class DetectedPlate:
    """Internal result; routes.py maps this onto the Plate schema."""

    box: tuple[int, int, int, int]
    detection_confidence: float
    reading: dict | None


@dataclass(slots=True)
class DetectionOutcome:
    annotated: np.ndarray
    plates: list[DetectedPlate]
    width: int
    height: int
    elapsed_ms: float


def decode_image(payload: bytes) -> np.ndarray:
    """
    Decode raw bytes into a BGR image.

    Raises ImageDecodeError rather than returning None, so callers cannot
    silently proceed with a null frame. Note that ultralytics patches
    cv2.imread to route through imdecode and *raises* on empty buffers, which
    is why decoding is done explicitly here instead of via a file path.
    """
    if not payload:
        raise ImageDecodeError('empty file')
    buffer = np.frombuffer(payload, dtype=np.uint8)
    if buffer.size == 0:
        raise ImageDecodeError('empty file')
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError('not a decodable image')
    return image


class PlateRecognizer:
    """Lazily-loaded detector + OCR pair."""

    def __init__(self, settings: Settings, backend: OCRBackend | None = None) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model = None
        # Injectable so a test or benchmark can supply a different engine
        # without touching configuration.
        self._ocr = backend or create_backend(settings)
        self._ocr_ready = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def ocr_loaded(self) -> bool:
        return self._ocr_ready

    @property
    def ocr_engine(self) -> str:
        return self._ocr.name

    @property
    def ready(self) -> bool:
        return self.model_loaded and self.ocr_loaded

    def load(self) -> None:
        """Load detector and OCR engine. Called once at startup; blocking."""
        model_path = Path(self._settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f'Model weights not found at {model_path}. '
                'See README "Model weights".'
            )

        from ultralytics import YOLO  # imported here to keep startup errors local

        started = time.monotonic()
        self._model = YOLO(str(model_path))
        logger.info('YOLO loaded from %s in %.2fs',
                    model_path, time.monotonic() - started)

        self._ocr.load()
        self._ocr_ready = True

    def close(self) -> None:
        self._model = None
        self._ocr.close()
        self._ocr_ready = False

    # -- inference ---------------------------------------------------------

    def _read_text(self, image: np.ndarray):
        return self._ocr.read(image)

    def detect(self, image: np.ndarray, *, max_passes: int | None = None,
               annotate: bool = True) -> DetectionOutcome:
        """
        Run detection + OCR over one BGR image.

        Blocking and CPU-bound. Callers on the async path must dispatch this to
        a thread. Returns an annotated copy plus per-plate readings.
        """
        if not self.ready:
            raise RuntimeError('models are not loaded')

        settings = self._settings
        started = time.monotonic()

        frame = self._fit_for_inference(image)
        height, width = frame.shape[:2]
        passes = settings.ocr_max_passes if max_passes is None else max_passes

        with self._lock:
            prediction = self._model.predict(
                source=frame,
                conf=settings.detection_confidence,
                verbose=False,
            )[0]

            plates: list[DetectedPlate] = []
            for box in prediction.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                det_conf = float(box.conf[0]) * 100

                # Pad proportionally to the plate's size. YOLO boxes hug the
                # glyphs, so a fixed margin clips the outermost character on
                # larger plates -- which silently shortens the reading.
                pad = max(
                    settings.crop_padding_min,
                    round(settings.crop_padding_ratio * max(x2 - x1, y2 - y1)),
                )
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(width, x2 + pad), min(height, y2 + pad)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                reading = read_plate_cascade(
                    crop,
                    self._read_text,
                    min_confidence=settings.ocr_min_confidence,
                    accept_confidence=settings.ocr_accept_confidence,
                    max_passes=passes,
                    time_budget=settings.ocr_time_budget,
                )
                plates.append(DetectedPlate((x1, y1, x2, y2), det_conf, reading))

        annotated = self._annotate(frame, plates) if annotate else frame
        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info('detected %d plate(s) in %.0fms', len(plates), elapsed_ms)

        return DetectionOutcome(annotated, plates, width, height, elapsed_ms)

    # -- helpers -----------------------------------------------------------

    def _fit_for_inference(self, image: np.ndarray) -> np.ndarray:
        """Downscale so the longest edge fits the configured budget."""
        limit = self._settings.max_inference_edge
        height, width = image.shape[:2]
        longest = max(height, width)
        if limit <= 0 or longest <= limit:
            return image
        scale = limit / longest
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        logger.debug('resized %dx%d -> %dx%d for inference',
                     width, height, resized.shape[1], resized.shape[0])
        return resized

    @staticmethod
    def _annotate(frame: np.ndarray, plates: list[DetectedPlate]) -> np.ndarray:
        """Draw boxes and labels on a copy, never on the caller's array."""
        canvas = frame.copy()
        for plate in plates:
            x1, y1, x2, y2 = plate.box
            reading = plate.reading

            if reading is None:
                label, colour = 'Not detected', (0, 165, 255)   # amber
            elif reading['valid']:
                label = f"{reading['number']} ({reading['confidence']:.0f}%)"
                colour = (0, 200, 0)                            # green
            else:
                # Trailing '?' marks a reading with no matching plate format.
                label = f"{reading['number']} ({reading['confidence']:.0f}%) ?"
                colour = (0, 165, 255)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

            scale, thickness = 0.6, 2
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )
            # Keep the label on-canvas when the box sits near the top edge.
            top = y1 - text_h - baseline - 4
            if top < 0:
                top = min(y2 + 4, canvas.shape[0] - text_h - baseline - 4)
            top = max(0, top)
            left = min(x1, max(0, canvas.shape[1] - text_w - 8))

            cv2.rectangle(
                canvas,
                (left, top),
                (left + text_w + 8, top + text_h + baseline + 4),
                colour, -1,
            )
            cv2.putText(
                canvas, label, (left + 4, top + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness,
                cv2.LINE_AA,
            )
        return canvas


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode('.jpg', image,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('failed to encode JPEG')
    return buffer.tobytes()
