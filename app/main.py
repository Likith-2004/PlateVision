"""
Application factory.

Models are loaded in the lifespan handler rather than at import time, so the
process starts, /health answers immediately, and a missing or corrupt weights
file surfaces as a reported error instead of an import-time crash.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import Settings, get_settings
from app.inference import PlateRecognizer
from app.routes import router
from app.storage import ArtifactStore

WEB_DIR = Path(__file__).resolve().parent / 'web'

DESCRIPTION = """
Detects Indian vehicle number plates with a fine-tuned YOLO model and reads
them with EasyOCR, then resolves the raw text against real Indian plate formats.

**Readings carry a `valid` flag.** `true` means the text matched a known format
(1988 series or Bharat series) after letter/digit confusion repair. `false`
means OCR produced something unrecognisable, and the text is returned as-is
rather than being presented as a confirmed plate.
"""


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.ensure_dirs()

    app.state.store = ArtifactStore(settings.detected_dir, settings.retention_hours)
    recognizer = PlateRecognizer(settings)
    app.state.recognizer = recognizer
    app.state.startup_error = None

    logger = logging.getLogger(__name__)

    async def warm_up() -> None:
        try:
            # Blocks for ~10s; keep it off the event loop.
            await run_in_threadpool(recognizer.load)
        except Exception as exc:
            # Stay up and report through /health rather than dying at boot, so
            # the cause is visible instead of buried in a restart loop.
            app.state.startup_error = str(exc)
            logger.error('model load failed: %s', exc)
        else:
            await run_in_threadpool(app.state.store.prune)

    # Loading happens in the background rather than being awaited here: uvicorn
    # does not bind the socket until lifespan startup returns, so awaiting it
    # would leave the port closed for the whole load and make /health's
    # "loading" state unobservable. Requests arriving early get a clean 503.
    loader = asyncio.create_task(warm_up())
    app.state.loader = loader

    try:
        yield
    finally:
        if not loader.done():
            loader.cancel()
        with suppress(asyncio.CancelledError):
            await loader
        recognizer.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    # Docs and schema move with the routes. Otherwise mounting the API under
    # /api would leave /docs at the root, where the deployment's routing table
    # sends it to the frontend service and it renders as a 404 page.
    prefix = settings.api_prefix

    app = FastAPI(
        title='PlateVision',
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url=f'{prefix}/docs',
        redoc_url=None,
        openapi_url=f'{prefix}/openapi.json',
        openapi_tags=[
            {'name': 'detection', 'description': 'Plate detection and recognition.'},
            {'name': 'system', 'description': 'Health and readiness.'},
        ],
    )
    app.state.settings = settings

    # Cross-origin access is opt-in. Needed only for a split deploy (UI on a
    # static host, API elsewhere); a single-host deploy needs none of this.
    if settings.cors_origins:
        allow_any = '*' in settings.cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'] if allow_any else settings.cors_origins,
            # Credentials cannot be combined with a wildcard origin, and this
            # API has no cookies or auth to send anyway.
            allow_credentials=False,
            allow_methods=['GET', 'POST', 'OPTIONS'],
            allow_headers=['*'],
            max_age=3600,
        )
        logging.getLogger(__name__).info(
            'CORS enabled for: %s', ', '.join(settings.cors_origins)
        )

    app.include_router(router, prefix=prefix)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Normalise every error into the ErrorResponse shape."""
        detail = exc.detail
        if isinstance(detail, dict) and 'error' in detail:
            body = detail
        else:
            body = {'error': 'http_error', 'detail': str(detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={'error': 'validation_error', 'detail': str(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        """Log the detail server-side; never leak internals to the client."""
        logging.getLogger(__name__).exception('unhandled error: %s', exc)
        return JSONResponse(
            status_code=500,
            content={'error': 'internal_error',
                     'detail': 'The server failed to process the request.'},
        )

    # app/web/ is the built React bundle (source lives in frontend/, built with
    # `npm --prefix frontend run build`). Absent on a source-only checkout that
    # has not been built yet, in which case the API still serves normally.
    index_file = WEB_DIR / 'index.html'
    assets_dir = WEB_DIR / 'assets'

    if assets_dir.is_dir():
        app.mount('/assets', StaticFiles(directory=assets_dir), name='assets')

    if index_file.is_file():
        @app.get('/', include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(index_file)
    else:
        @app.get('/', include_in_schema=False)
        async def missing_ui() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    'error': 'ui_not_built',
                    'detail': 'Run: npm --prefix frontend install '
                              '&& npm --prefix frontend run build. '
                              'The API itself is available at /docs.',
                },
            )

    return app


app = create_app()
