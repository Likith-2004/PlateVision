"""
Typed application settings.

Every knob is read from the environment (or a .env file) exactly once at
import time and validated by pydantic, so a bad value fails at startup with a
clear message instead of surfacing as a confusing runtime error later.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# pydantic-settings JSON-decodes environment values for collection fields before
# any validator runs, so `OCR_LANGUAGES=en` raises SettingsError rather than
# reaching the parsing below. NoDecode turns that off and hands the raw string
# to the `mode='before'` validators, which accept both plain and JSON forms.
CsvList = Annotated[list[str], NoDecode]
CsvSet = Annotated[set[str], NoDecode]


def _as_items(value):
    """
    Normalise a collection setting into a list of strings.

    Accepts what a human would put in a .env file -- "en", "en,hi", "en hi" --
    and also a JSON array, which is pydantic-settings' native form and the thing
    someone is most likely to try if the plain form ever looks ambiguous.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text.startswith('[') and text.endswith(']'):
        try:
            decoded = json.loads(text)
        except ValueError:
            pass
        else:
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]

    return [part.strip() for part in text.replace(',', ' ').split() if part.strip()]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # -- Server ------------------------------------------------------------
    host: str = '127.0.0.1'
    port: int = 8000
    reload: bool = False
    log_level: str = 'info'

    # Browser origins allowed to call the API. Only needed when the frontend is
    # served from a different host than the API. Empty means same-origin only,
    # which is correct for a single deployment -- including the Vercel one,
    # where the UI and the API sit behind one domain.
    # Accepts a comma-separated list, or "*" to allow any origin.
    cors_origins: CsvList = Field(default_factory=list)

    # Path prefix every API route is mounted under. Empty by default, so a
    # local run keeps /health and /detect at the root.
    #
    # Set to "/api" for the Vercel deployment: a Vercel service receives the
    # ORIGINAL request path, so a rewrite of /api/(.*) to this service arrives
    # as "/api/detect", not "/detect". Without the matching prefix here every
    # API call 404s while the UI itself loads perfectly -- a confusing failure
    # worth making impossible.
    api_prefix: str = ''

    # -- Paths -------------------------------------------------------------
    model_path: Path = PROJECT_ROOT / 'model' / 'best.pt'
    data_dir: Path = PROJECT_ROOT / 'data'

    # -- Uploads -----------------------------------------------------------
    max_upload_mb: int = 10
    allowed_extensions: CsvSet = Field(
        default_factory=lambda: {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    )
    # Annotated outputs are disposable; prune anything older than this so a
    # long-running demo cannot fill the disk. 0 disables pruning.
    retention_hours: int = 24

    # Return the annotated image inline in the JSON response as a data URI
    # instead of writing it to disk and returning a /media/... URL.
    #
    # Required on serverless hosts, where this is a correctness fix rather than
    # a preference: the filesystem is ephemeral AND per-instance, so the
    # browser's follow-up GET /media/<id>.jpg can land on a different instance
    # that never wrote the file, yielding an intermittently broken image. The
    # cost is ~33% base64 overhead on one response; the annotated JPEG is
    # normally well under 1 MB, and inlining also removes a round trip.
    inline_images: bool = False

    # -- Detection ---------------------------------------------------------
    # How sure YOLO must be that a region IS a number plate.
    detection_confidence: float = 0.5
    # Longest edge fed to YOLO. Larger images are downscaled first.
    max_inference_edge: int = 1280

    # Crop padding around a detected plate, as a fraction of the box's longer
    # side (with a pixel floor). Proportional rather than fixed because YOLO
    # boxes sit tight against the glyphs: a constant 10px clipped the trailing
    # character of a Bharat-series plate, and two of five preprocessing
    # variants then read "22BH6517" instead of "22BH6517A". At ~6% every
    # variant recovers the full plate.
    crop_padding_ratio: float = 0.06
    crop_padding_min: int = 8

    # -- OCR ---------------------------------------------------------------
    # paddleocr reads the sample plates at ~99% in one pass and ~4-8x faster
    # than easyocr; see app/ocr.py for the measurements behind this default.
    ocr_engine: Literal['paddleocr', 'easyocr'] = 'paddleocr'
    ocr_languages: CsvList = Field(default_factory=lambda: ['en'])
    ocr_gpu: bool = False
    # Mobile tier, pinned deliberately: PaddleOCR 3.x otherwise picks
    # server-grade models that are ~40x slower on CPU for no accuracy gain.
    paddle_detection_model: str = 'PP-OCRv5_mobile_det'
    paddle_recognition_model: str = 'PP-OCRv5_mobile_rec'

    # Where PaddleOCR's weights live. Both unset by default, which leaves
    # Paddle's own ~/.paddlex behaviour untouched -- correct for a local run.
    #
    # A read-only or ephemeral host needs both: `paddle_model_dir` is the
    # copy baked into the deployment by scripts/warm_paddle.py, and
    # `paddle_cache_dir` is a writable location seeded from it at load time.
    # Without them PaddleOCR tries to download ~21 MB into a read-only $HOME on
    # every cold start.
    paddle_model_dir: Path | None = None
    paddle_cache_dir: Path | None = None
    # Floor for accepting OCR text that does NOT match a known plate format.
    # Format-valid readings bypass this entirely.
    ocr_min_confidence: float = 0.1
    # Confidence at which a format-valid reading ends the cascade immediately.
    ocr_accept_confidence: float = 0.70
    # Preprocessing passes per plate. Pass 1 is cheap and native-resolution;
    # later passes upscale (~4x the pixels) and only run when pass 1 fails.
    ocr_max_passes: int = 3
    # Wall-clock ceiling in seconds for OCR of a single plate. The first pass
    # always runs; escalation stops once this is exceeded.
    ocr_time_budget: float = 4.0
    # Passes allowed for live/streaming frames, where latency dominates.
    ocr_live_max_passes: int = 1

    # -- Concurrency -------------------------------------------------------
    # Neither YOLO nor EasyOCR is thread-safe, and CPU inference does not
    # benefit from oversubscription, so requests queue on a small pool.
    inference_workers: int = 1

    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def _normalise_extensions(cls, value):
        """Accept "jpg,png" or {".jpg"}; always store lowercase with a dot."""
        return {
            ('.' + str(ext).lstrip('.')).lower()
            for ext in _as_items(value)
        }

    @field_validator('ocr_languages', 'cors_origins', mode='before')
    @classmethod
    def _split_items(cls, value):
        """Accept "https://a.app,https://b.app" as well as a real list."""
        return _as_items(value)

    @field_validator('model_path', 'data_dir', 'paddle_model_dir',
                     'paddle_cache_dir')
    @classmethod
    def _anchor_to_project(cls, value: Path | None) -> Path | None:
        """
        Resolve a relative path against the project root, not the CWD.

        Hosts are inconsistent about the working directory a process starts in,
        so `PADDLE_MODEL_DIR=.paddlex` must not mean "wherever this happened to
        be launched from" -- that produces a cache miss and a surprise download
        rather than an error.
        """
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @field_validator('api_prefix', mode='before')
    @classmethod
    def _normalise_prefix(cls, value):
        """
        Accept "api", "/api" or "/api/" and always store "/api".

        FastAPI silently accepts a prefix without a leading slash and then
        matches nothing, so normalising here turns a plausible typo into
        working configuration instead of a wholly 404 API.
        """
        text = str(value or '').strip().strip('/')
        return f'/{text}' if text else ''

    @field_validator('detection_confidence', 'ocr_min_confidence',
                     'ocr_accept_confidence')
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError('must be between 0.0 and 1.0')
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / 'uploads'

    @property
    def detected_dir(self) -> Path:
        return self.data_dir / 'detected'

    def ensure_dirs(self) -> None:
        for path in (self.upload_dir, self.detected_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
