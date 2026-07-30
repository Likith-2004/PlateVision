"""
Behaviours the hosted deployment depends on.

Everything here is invisible on a local run and fatal in production, which is
exactly the combination that does not get caught by hand:

* API_PREFIX -- a Vercel service receives the ORIGINAL request path, so a
  rewrite of /api/* arrives as "/api/detect". Mount the router at the root and
  the UI loads perfectly while every single call 404s.
* INLINE_IMAGES -- serverless instances have their own ephemeral /tmp, so the
  browser's follow-up GET for an annotated image can reach an instance that
  never wrote it. The bug is intermittent, which is worse than broken.
* The Paddle weight cache -- absent, PaddleOCR downloads ~21 MB into a
  read-only $HOME on every cold start.
"""

from __future__ import annotations

import base64
import shutil

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import PROJECT_ROOT, StubRecognizer, make_plate


def build(settings, monkeypatch, plates=None):
    """A TestClient over `settings` with the model layer stubbed out."""
    stub = StubRecognizer(plates=plates if plates is not None else [make_plate()])

    import app.main as main_module

    monkeypatch.setattr(main_module, 'PlateRecognizer', lambda _s: stub)
    client = TestClient(create_app(settings))
    client.stub = stub
    return client


# --------------------------------------------------------------- api_prefix

@pytest.mark.parametrize('raw,expected', [
    ('/api', '/api'),
    ('api', '/api'),          # a plausible typo that would otherwise 404 all
    ('/api/', '/api'),
    ('api/', '/api'),
    ('', ''),
    ('/', ''),
])
def test_api_prefix_is_normalised(raw, expected):
    assert Settings(_env_file=None, api_prefix=raw).api_prefix == expected


def test_routes_move_under_the_prefix(tmp_path, monkeypatch, png_bytes):
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd', api_prefix='/api')

    with build(settings, monkeypatch) as client:
        assert client.get('/api/health').status_code == 200
        posted = client.post(
            '/api/detect', files={'image': ('car.png', png_bytes, 'image/png')}
        )
        assert posted.status_code == 200
        assert posted.json()['plates'][0]['number'] == 'MH12AB1234'


def test_unprefixed_routes_are_gone_when_a_prefix_is_set(tmp_path, monkeypatch):
    """
    Guards against mounting the router twice.

    If /health answered at both paths the prefix would look fine locally and
    still be wrong, because only one of them is reachable through the
    deployment's routing table.
    """
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd', api_prefix='/api')

    with build(settings, monkeypatch) as client:
        assert client.get('/health').status_code == 404


def test_docs_and_schema_follow_the_prefix(tmp_path, monkeypatch):
    """
    The UI links to `${API_BASE}/docs`, and /docs at the root would be routed to
    the frontend service instead of here.
    """
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd', api_prefix='/api')

    with build(settings, monkeypatch) as client:
        assert client.get('/api/openapi.json').status_code == 200
        assert client.get('/api/docs').status_code == 200
        assert client.get('/openapi.json').status_code == 404


def test_media_url_carries_the_prefix(tmp_path, monkeypatch, png_bytes):
    """A returned URL must be fetchable as-is, prefix included."""
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd', api_prefix='/api')

    with build(settings, monkeypatch) as client:
        body = client.post(
            '/api/detect', files={'image': ('car.png', png_bytes, 'image/png')}
        ).json()

        assert body['image_url'].startswith('/api/media/')
        assert client.get(body['image_url']).status_code == 200


def test_default_deploy_keeps_routes_at_the_root(client):
    """The single-host default must be untouched by any of the above."""
    assert client.get('/health').status_code == 200
    assert client.get('/openapi.json').status_code == 200


# ------------------------------------------------------------ inline_images

def test_inline_images_returns_a_data_uri(tmp_path, monkeypatch, png_bytes):
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd', inline_images=True)

    with build(settings, monkeypatch) as client:
        body = client.post(
            '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
        ).json()

    url = body['image_url']
    assert url.startswith('data:image/jpeg;base64,')

    payload = base64.b64decode(url.split(',', 1)[1], validate=True)
    assert payload[:2] == b'\xff\xd8'                    # JPEG magic
    assert payload[-2:] == b'\xff\xd9'                   # complete image


def test_inline_images_writes_nothing_to_disk(tmp_path, monkeypatch, png_bytes):
    """
    The point of inlining is that no artifact is needed. If a file still lands
    on disk the deployment would silently depend on a filesystem it does not
    reliably have.
    """
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd', inline_images=True)

    with build(settings, monkeypatch) as client:
        client.post('/detect', files={'image': ('car.png', png_bytes, 'image/png')})

    assert list(settings.detected_dir.glob('*.jpg')) == []


def test_disk_mode_remains_the_default(tmp_path, monkeypatch, png_bytes):
    settings = Settings(_env_file=None, data_dir=tmp_path / 'd')
    assert settings.inline_images is False

    with build(settings, monkeypatch) as client:
        body = client.post(
            '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
        ).json()

    assert body['image_url'].startswith('/media/')
    assert len(list(settings.detected_dir.glob('*.jpg'))) == 1


# ------------------------------------------------------------- paddle cache

def test_relative_paddle_dirs_anchor_to_the_project(monkeypatch):
    """
    `PADDLE_MODEL_DIR=.paddlex` must mean the project directory, not whatever
    directory the host happened to launch the process from -- otherwise the
    bundled weights are missed and Paddle silently downloads them instead.
    """
    monkeypatch.setenv('PADDLE_MODEL_DIR', '.paddlex')
    monkeypatch.setenv('PADDLE_CACHE_DIR', '/tmp/pdx')

    settings = Settings(_env_file=None)
    assert settings.paddle_model_dir == PROJECT_ROOT / '.paddlex'
    # Absolute values are left alone.
    assert settings.paddle_cache_dir.is_absolute()


def test_paddle_dirs_default_to_unset():
    """Unset means "leave Paddle's own ~/.paddlex alone", which local runs want."""
    settings = Settings(_env_file=None)
    assert settings.paddle_model_dir is None
    assert settings.paddle_cache_dir is None


def test_cache_is_seeded_from_the_bundled_models(tmp_path, monkeypatch):
    """The runtime copy must avoid the network entirely."""
    from app.ocr import _prepare_cache

    seed = tmp_path / 'bundled'
    weights = seed / 'official_models' / 'PP-OCRv5_mobile_det'
    weights.mkdir(parents=True)
    (weights / 'inference.pdmodel').write_bytes(b'weights')

    cache = tmp_path / 'writable'
    monkeypatch.delenv('PADDLE_PDX_CACHE_HOME', raising=False)

    assert _prepare_cache(seed, cache) == cache
    copied = cache / 'official_models' / 'PP-OCRv5_mobile_det' / 'inference.pdmodel'
    assert copied.read_bytes() == b'weights'
    # Paddle reads this at import time; without it the copy is pointless.
    import os
    assert os.environ['PADDLE_PDX_CACHE_HOME'] == str(cache)


def test_seeding_is_idempotent_and_keeps_existing_files(tmp_path, monkeypatch):
    """A warm instance must not re-copy over models already in place."""
    from app.ocr import _prepare_cache

    seed = tmp_path / 'bundled'
    model = seed / 'official_models' / 'PP-OCRv5_mobile_rec'
    model.mkdir(parents=True)
    (model / 'w').write_bytes(b'fresh')

    cache = tmp_path / 'writable'
    _prepare_cache(seed, cache)
    (cache / 'official_models' / 'PP-OCRv5_mobile_rec' / 'w').write_bytes(b'kept')
    _prepare_cache(seed, cache)

    assert (cache / 'official_models' / 'PP-OCRv5_mobile_rec' / 'w').read_bytes() == b'kept'


def test_no_cache_dir_leaves_paddle_defaults_alone(tmp_path, monkeypatch):
    from app.ocr import _prepare_cache

    monkeypatch.delenv('PADDLE_PDX_CACHE_HOME', raising=False)
    assert _prepare_cache(tmp_path, None) is None

    import os
    assert 'PADDLE_PDX_CACHE_HOME' not in os.environ


def test_unwritable_cache_falls_back_instead_of_crashing(tmp_path, monkeypatch):
    """
    A permissions problem should degrade to "download it" rather than take the
    whole service down at startup.
    """
    from app import ocr

    monkeypatch.delenv('PADDLE_PDX_CACHE_HOME', raising=False)
    monkeypatch.setattr(
        ocr.Path, 'mkdir',
        lambda *_a, **_k: (_ for _ in ()).throw(OSError('read-only file system')),
    )
    assert ocr._prepare_cache(tmp_path, tmp_path / 'nope') is None


# ------------------------------------------------------------- build scripts

def test_fetch_model_is_a_noop_when_weights_exist(tmp_path, monkeypatch):
    """Local runs and repeat builds must never touch the network."""
    import runpy

    weights = tmp_path / 'best.pt'
    weights.write_bytes(b'x' * 2048)

    monkeypatch.setenv('MODEL_PATH', str(weights))
    monkeypatch.delenv('MODEL_SHA256', raising=False)
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda *_a, **_k: pytest.fail('fetch_model hit the network'),
    )

    runpy.run_path(str(PROJECT_ROOT / 'scripts' / 'fetch_model.py'),
                   run_name='__main__')


def test_fetch_model_fails_loudly_without_a_url(tmp_path, monkeypatch):
    """
    A missing model must fail the BUILD. Deferring it turns a clear build error
    into a deployment that starts, serves the UI, and reports a model error
    through /health forever.
    """
    import runpy

    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'absent.pt'))
    monkeypatch.delenv('MODEL_URL', raising=False)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(PROJECT_ROOT / 'scripts' / 'fetch_model.py'),
                       run_name='__main__')
    assert excinfo.value.code == 1


def test_fetch_model_rejects_a_checksum_mismatch(tmp_path, monkeypatch):
    import runpy

    weights = tmp_path / 'best.pt'
    weights.write_bytes(b'x' * 2048)
    monkeypatch.setenv('MODEL_PATH', str(weights))
    monkeypatch.setenv('MODEL_SHA256', 'deadbeef')

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(PROJECT_ROOT / 'scripts' / 'fetch_model.py'),
                       run_name='__main__')
    assert excinfo.value.code == 1


def test_fetch_model_rejects_an_implausibly_small_download(tmp_path, monkeypatch):
    """
    A few KB means the URL served an error page, an HTML redirect or a Git LFS
    pointer. Accepting it would surface much later inside ultralytics.
    """
    import io
    import runpy

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'best.pt'))
    monkeypatch.setenv('MODEL_URL', 'https://example.invalid/best.pt')
    monkeypatch.delenv('MODEL_SHA256', raising=False)
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda *_a, **_k: Response(b'<html>404</html>'))

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(PROJECT_ROOT / 'scripts' / 'fetch_model.py'),
                       run_name='__main__')
    assert excinfo.value.code == 1
    assert not (tmp_path / 'best.pt').exists()
    assert not (tmp_path / 'best.pt.part').exists()


# ------------------------------------------------------------------ config

def test_vercel_config_matches_the_backend_contract():
    """
    vercel.json and the app must agree, since a mismatch produces a UI that
    loads and an API that 404s -- with nothing in the logs to explain it.
    """
    import json

    config = json.loads((PROJECT_ROOT / 'vercel.json').read_text())

    api = config['services']['api']
    assert api['entrypoint'] == 'app.main:app'
    # excludeFiles is keyed by the RESOLVED entrypoint file, not the module path.
    assert 'app/main.py' in api['functions']

    # The rewrite prefix, the frontend's VITE_API_BASE and the backend's
    # API_PREFIX are three spellings of one value.
    sources = [rule['source'] for rule in config['rewrites']]
    assert '/api/(.*)' in sources
    assert sources[-1] == '/(.*)', 'the catch-all must be matched last'

    env = (PROJECT_ROOT / 'frontend' / '.env.vercel').read_text()
    assert 'VITE_API_BASE=/api' in env

    # Weights and Paddle models must survive bundling.
    excluded = api['functions']['app/main.py']['excludeFiles']
    assert 'model/**' not in excluded
    assert '.paddlex' not in excluded


def test_deploy_requirements_omit_the_cuda_torch_build():
    """
    Plain `torch` on Linux pulls ~2 GB of CUDA wheels this project never uses,
    which alone would blow the function bundle limit.
    """
    text = (PROJECT_ROOT / 'requirements-deploy.txt').read_text()
    assert 'download.pytorch.org/whl/cpu' in text
    assert 'torch==2.7.1+cpu' in text

    # Comments discuss easyocr as an option; only the requirement lines matter.
    pinned = [
        line.split('#', 1)[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    assert not any(line.startswith('easyocr') for line in pinned)
    assert not any(line.startswith('torch==2.7.1\n') for line in pinned)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_frontend_vercel_build_uses_the_vercel_mode():
    """Without --mode vercel the bundle is built with an empty API base."""
    import json

    package = json.loads((PROJECT_ROOT / 'frontend' / 'package.json').read_text())
    assert '--mode vercel' in package['scripts']['build:vercel']
