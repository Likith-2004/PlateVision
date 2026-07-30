"""
API response models.

These are the contract. FastAPI derives the OpenAPI schema at /docs from them,
so the documented shape cannot drift from what the code actually returns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Plate location in pixels, relative to the returned annotated image."""

    model_config = ConfigDict(frozen=True)

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class Plate(BaseModel):
    """One detected plate and its best reading."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            'example': {
                'number': '22BH6517',
                'confidence': 81.68,
                'detection_confidence': 83.8,
                'valid': True,
                'format': 'bh_series',
                'state': None,
                'corrections': 0,
                'variant': 'clahe',
                'passes': 1,
                'votes': 1,
                'box': {'x1': 49, 'y1': 39, 'x2': 389, 'y2': 160},
            }
        },
    )

    number: str = Field(description='Best plate reading, uppercase alphanumeric.')
    confidence: float = Field(
        ge=0, le=100, description='OCR confidence for this reading, percent.'
    )
    detection_confidence: float = Field(
        ge=0, le=100, description="YOLO's confidence that this region is a plate."
    )
    valid: bool = Field(
        description='True when the reading matches a known Indian plate format. '
                    'False means the text is returned as-is and is unreliable.'
    )
    format: Literal['standard', 'bh_series'] | None = Field(
        default=None, description='Matched plate format, null when invalid.'
    )
    state: str | None = Field(
        default=None, description='State/UT code for standard-format plates.'
    )
    corrections: int = Field(
        default=0, ge=0,
        description='Characters repaired via letter/digit confusion rules.'
    )
    variant: str | None = Field(
        default=None, description='Preprocessing variant that produced the reading.'
    )
    passes: int = Field(default=1, ge=0, description='OCR passes spent on this plate.')
    votes: int = Field(
        default=1, ge=0,
        description='Independent preprocessing variants that agreed on this number.'
    )
    box: BoundingBox


class DetectionResult(BaseModel):
    """Outcome of running the pipeline over one image."""

    plates: list[Plate] = Field(
        default_factory=list,
        description='Empty when no plate region was detected at all.',
    )
    image_url: str = Field(description='Annotated image, served under /media.')
    source_width: int
    source_height: int
    elapsed_ms: float = Field(description='Server-side pipeline wall time.')


class HealthStatus(BaseModel):
    """Readiness probe. `ready` is false until the models finish loading."""

    status: Literal['ok', 'loading', 'error']
    ready: bool
    model_loaded: bool
    ocr_loaded: bool
    ocr_engine: str = Field(description='Active OCR backend.')
    version: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error body for every non-2xx response."""

    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'error': 'unsupported_media_type',
                'detail': 'Unsupported file type ".md".',
                'allowed': ['.bmp', '.jpeg', '.jpg', '.png', '.webp'],
            }
        }
    )

    error: str = Field(description='Stable machine-readable code.')
    detail: str = Field(description='Human-readable explanation.')
    allowed: list[str] | None = None
