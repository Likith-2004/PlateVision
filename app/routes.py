"""
HTTP API.

Inference is CPU-bound and blocking, so every call is dispatched to a worker
thread via `run_in_threadpool`. That keeps the event loop responsive: a request
spending 5 seconds in OCR does not stop the server from accepting others, and
/health stays answerable while a detection is in flight.
"""

from __future__ import annotations

import base64
import logging
from pathlib import PurePosixPath

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.inference import ImageDecodeError, PlateRecognizer, decode_image, encode_jpeg
from app.schemas import (
    BoundingBox,
    DetectionResult,
    ErrorResponse,
    HealthStatus,
    Plate,
)
from app.storage import ArtifactStore

logger = logging.getLogger(__name__)

router = APIRouter()

ERROR_RESPONSES: dict[int | str, dict] = {
    400: {'model': ErrorResponse, 'description': 'Malformed or undecodable image'},
    413: {'model': ErrorResponse, 'description': 'Upload exceeds the size limit'},
    415: {'model': ErrorResponse, 'description': 'Unsupported file extension'},
    503: {'model': ErrorResponse, 'description': 'Models still loading'},
}


def _data_uri(payload: bytes) -> str:
    """Encode an annotated JPEG for direct use as an <img> src."""
    return 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')


def _to_plate(detected) -> Plate:
    """Map an internal DetectedPlate onto the public schema."""
    reading = detected.reading or {}
    return Plate(
        number=reading.get('number', 'Not detected'),
        confidence=reading.get('confidence', 0.0),
        detection_confidence=round(detected.detection_confidence, 2),
        valid=bool(reading.get('valid', False)),
        format=reading.get('format'),
        state=reading.get('state'),
        corrections=reading.get('corrections', 0),
        variant=reading.get('variant'),
        passes=reading.get('passes', 0),
        votes=reading.get('votes', 0),
        box=BoundingBox(x1=detected.box[0], y1=detected.box[1],
                        x2=detected.box[2], y2=detected.box[3]),
    )


def _recognizer(request: Request) -> PlateRecognizer:
    recognizer: PlateRecognizer = request.app.state.recognizer
    if not recognizer.ready:
        raise HTTPException(
            status_code=503,
            detail={'error': 'not_ready',
                    'detail': 'Models are still loading; retry shortly.'},
        )
    return recognizer


async def _read_upload(request: Request, upload: UploadFile) -> bytes:
    """Validate extension and size, then return the bytes."""
    settings = request.app.state.settings

    suffix = PurePosixPath(upload.filename or '').suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail={
                'error': 'unsupported_media_type',
                'detail': f'Unsupported file type "{suffix or "unknown"}".',
                'allowed': sorted(settings.allowed_extensions),
            },
        )

    # Read with a hard cap. Streaming in chunks means an oversized body is
    # rejected without ever being held in memory in full.
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail={
                    'error': 'payload_too_large',
                    'detail': f'File exceeds the {settings.max_upload_mb} MB limit.',
                },
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(
            status_code=400,
            detail={'error': 'empty_file', 'detail': 'The uploaded file is empty.'},
        )
    return b''.join(chunks)


@router.get('/health', response_model=HealthStatus, tags=['system'])
async def health(request: Request) -> HealthStatus:
    """Readiness probe. Returns 200 with ready=false while models load."""
    recognizer: PlateRecognizer = request.app.state.recognizer
    error: str | None = getattr(request.app.state, 'startup_error', None)

    engine = getattr(recognizer, 'ocr_engine', 'unknown')

    if error:
        return HealthStatus(status='error', ready=False,
                            model_loaded=recognizer.model_loaded,
                            ocr_loaded=recognizer.ocr_loaded,
                            ocr_engine=engine,
                            version=__version__, detail=error)

    ready = recognizer.ready
    return HealthStatus(
        status='ok' if ready else 'loading',
        ready=ready,
        model_loaded=recognizer.model_loaded,
        ocr_loaded=recognizer.ocr_loaded,
        ocr_engine=engine,
        version=__version__,
    )


@router.post('/detect', response_model=DetectionResult,
             responses=ERROR_RESPONSES, tags=['detection'])
async def detect(
    request: Request,
    background: BackgroundTasks,
    image: UploadFile = File(description='Image containing one or more plates.'),
) -> DetectionResult:
    """
    Detect number plates in an uploaded image.

    Returns every detected plate with its best reading. A reading with
    `valid: false` did not match a known Indian plate format and should be
    treated as unreliable rather than displayed as a confirmed result.
    """
    recognizer = _recognizer(request)
    payload = await _read_upload(request, image)

    try:
        frame = await run_in_threadpool(decode_image, payload)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={'error': 'undecodable_image',
                    'detail': f'Could not decode the image: {exc}.'},
        ) from exc

    outcome = await run_in_threadpool(recognizer.detect, frame)

    settings = request.app.state.settings
    encoded = await run_in_threadpool(encode_jpeg, outcome.annotated)

    if settings.inline_images:
        # Serverless: no shared disk to write to, and no guarantee the follow-up
        # GET reaches this instance. Hand the image back in this response.
        image_url = await run_in_threadpool(_data_uri, encoded)
    else:
        store: ArtifactStore = request.app.state.store
        name = await run_in_threadpool(store.save_jpeg, encoded)
        image_url = f'{settings.api_prefix}/media/{name}'
        # Opportunistic cleanup, after the response is sent.
        background.add_task(store.prune)

    return DetectionResult(
        plates=[_to_plate(p) for p in outcome.plates],
        image_url=image_url,
        source_width=outcome.width,
        source_height=outcome.height,
        elapsed_ms=round(outcome.elapsed_ms, 1),
    )


@router.post('/detect/frame', response_model=DetectionResult,
             responses=ERROR_RESPONSES, tags=['detection'])
async def detect_frame(
    request: Request,
    image: UploadFile = File(description='Single frame from a live camera.'),
) -> DetectionResult:
    """
    Low-latency variant for live camera frames.

    Runs a single cheap OCR pass instead of the full cascade, and skips writing
    an artifact to disk: at frame rates the escalation cost is not worth paying
    and the annotated image would be discarded immediately anyway.

    The camera itself belongs in the browser. The previous implementation opened
    the *server's* camera, which is meaningless once deployed anywhere but the
    developer's own machine.
    """
    recognizer = _recognizer(request)
    payload = await _read_upload(request, image)

    try:
        frame = await run_in_threadpool(decode_image, payload)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={'error': 'undecodable_image',
                    'detail': f'Could not decode the frame: {exc}.'},
        ) from exc

    settings = request.app.state.settings
    outcome = await run_in_threadpool(
        recognizer.detect, frame,
        max_passes=settings.ocr_live_max_passes,
        annotate=False,
    )

    return DetectionResult(
        plates=[_to_plate(p) for p in outcome.plates],
        image_url='',
        source_width=outcome.width,
        source_height=outcome.height,
        elapsed_ms=round(outcome.elapsed_ms, 1),
    )


@router.get('/media/{name}', tags=['detection'],
            responses={404: {'model': ErrorResponse}})
async def media(request: Request, name: str) -> FileResponse:
    """Serve an annotated image produced by /detect."""
    store: ArtifactStore = request.app.state.store
    path = store.path_for(name)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={'error': 'not_found', 'detail': 'No such image.'},
        )
    return FileResponse(path, media_type='image/jpeg')
