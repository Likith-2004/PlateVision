"""
HTTP contract tests.

These run against a stubbed recognizer, so they cover validation, error
shape and status codes without loading the model. Every case here maps to a
failure the previous Flask implementation returned a 500 (plus traceback) for.
"""

from __future__ import annotations

import io

import pytest

from tests.conftest import make_plate


# ------------------------------------------------------------------- health

def test_health_reports_ready(client):
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['ready'] is True
    assert body['model_loaded'] is True
    assert body['version']


def test_index_is_served(client):
    """The built React shell is served at the root."""
    response = client.get('/')
    assert response.status_code == 200
    assert 'PlateVision' in response.text
    assert 'id="root"' in response.text


def test_built_assets_are_reachable(client):
    """
    The bundle referenced by index.html must actually resolve.

    A stale index.html pointing at a hashed filename that no longer exists
    yields a blank page in the browser with only a console error, so it is
    worth asserting rather than eyeballing.
    """
    import re

    from app.main import WEB_DIR

    if not (WEB_DIR / 'assets').is_dir():
        pytest.skip('frontend not built; run npm --prefix frontend run build')

    html = client.get('/').text
    refs = re.findall(r'/assets/[A-Za-z0-9._-]+\.(?:js|css)', html)
    assert refs, 'index.html references no built assets'

    for ref in refs:
        assert client.get(ref).status_code == 200, f'{ref} is missing'


def test_openapi_schema_is_generated(client):
    schema = client.get('/openapi.json').json()
    assert '/detect' in schema['paths']
    assert 'Plate' in schema['components']['schemas']


# ------------------------------------------------------------------- detect

def test_detect_returns_plates(client, png_bytes):
    response = client.post(
        '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body['plates']) == 1
    plate = body['plates'][0]
    assert plate['number'] == 'MH12AB1234'
    assert plate['valid'] is True
    assert plate['format'] == 'standard'
    assert plate['state'] == 'MH'
    assert plate['box'] == {'x1': 10, 'y1': 20, 'x2': 210, 'y2': 90}
    assert body['image_url'].startswith('/media/')
    assert body['elapsed_ms'] > 0


def test_detected_image_is_retrievable(client, png_bytes):
    body = client.post(
        '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
    ).json()

    media = client.get(body['image_url'])
    assert media.status_code == 200
    assert media.headers['content-type'] == 'image/jpeg'
    assert media.content[:2] == b'\xff\xd8'          # JPEG magic


def test_no_plates_is_a_success_with_an_empty_list(client, png_bytes):
    client.stub._plates = []
    body = client.post(
        '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
    ).json()
    assert body['plates'] == []


def test_unverified_plate_is_flagged_not_hidden(client, png_bytes):
    client.stub._plates = [make_plate('IHR26D0555J8', valid=False)]
    body = client.post(
        '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
    ).json()
    plate = body['plates'][0]
    assert plate['valid'] is False
    assert plate['format'] is None
    assert plate['number'] == 'IHR26D0555J8'


def test_uploads_do_not_collide(client, png_bytes):
    """Two uploads of the same filename must not overwrite each other."""
    first = client.post(
        '/detect', files={'image': ('same.png', png_bytes, 'image/png')}
    ).json()
    second = client.post(
        '/detect', files={'image': ('same.png', png_bytes, 'image/png')}
    ).json()

    assert first['image_url'] != second['image_url']
    assert client.get(first['image_url']).status_code == 200
    assert client.get(second['image_url']).status_code == 200


# -------------------------------------------------------------- error paths

def test_empty_file_is_400_not_500(client):
    """Previously: cv2.error traceback and HTTP 500."""
    response = client.post('/detect', files={'image': ('empty.jpg', b'', 'image/jpeg')})
    assert response.status_code == 400
    assert response.json()['error'] == 'empty_file'


def test_undecodable_image_is_400_not_500(client):
    """Bytes with an image extension that are not actually an image."""
    response = client.post(
        '/detect', files={'image': ('fake.jpg', b'this is not an image', 'image/jpeg')}
    )
    assert response.status_code == 400
    assert response.json()['error'] == 'undecodable_image'


def test_wrong_extension_is_415(client):
    """Previously: written to a public directory, then a 500."""
    response = client.post(
        '/detect', files={'image': ('notes.md', b'# hello', 'text/markdown')}
    )
    assert response.status_code == 415
    body = response.json()
    assert body['error'] == 'unsupported_media_type'
    assert '.jpg' in body['allowed']


def test_missing_field_is_422(client):
    response = client.post('/detect')
    assert response.status_code == 422
    assert response.json()['error'] == 'validation_error'


def test_oversized_upload_is_413(client):
    """max_upload_mb is 1 in tests."""
    payload = b'\xff\xd8' + b'x' * (2 * 1024 * 1024)
    response = client.post(
        '/detect', files={'image': ('big.jpg', payload, 'image/jpeg')}
    )
    assert response.status_code == 413
    assert response.json()['error'] == 'payload_too_large'


def test_errors_never_leak_internals(client):
    """Error bodies carry a stable code and prose, never a stack trace."""
    body = client.post(
        '/detect', files={'image': ('fake.jpg', b'nope', 'image/jpeg')}
    ).json()
    assert set(body) <= {'error', 'detail', 'allowed'}
    assert 'Traceback' not in body['detail']


@pytest.mark.parametrize('name', [
    'nope.jpg',
    '../../secret.jpg',
    'not-a-uuid.jpg',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png',   # wrong suffix
])
def test_media_rejects_bad_names(client, name):
    assert client.get(f'/media/{name}').status_code in (400, 404)


def test_media_path_traversal_is_blocked(client, tmp_path):
    response = client.get('/media/..%2F..%2Fconfig.py')
    assert response.status_code in (400, 404)
    assert 'Settings' not in response.text


# --------------------------------------------------------------- live frames

def test_frame_endpoint_uses_a_single_pass_and_skips_artifacts(client, png_bytes):
    response = client.post(
        '/detect/frame', files={'image': ('frame.jpg', png_bytes, 'image/jpeg')}
    )
    assert response.status_code == 200
    body = response.json()
    assert body['image_url'] == ''            # nothing written to disk
    assert body['plates'][0]['number'] == 'MH12AB1234'

    call = client.stub.calls[-1]
    assert call['max_passes'] == 1
    assert call['annotate'] is False


# ----------------------------------------------------------------- readiness

def test_requests_are_rejected_while_models_load(client, png_bytes):
    client.stub._ready = False
    response = client.post(
        '/detect', files={'image': ('car.png', png_bytes, 'image/png')}
    )
    assert response.status_code == 503
    assert response.json()['error'] == 'not_ready'


def test_health_still_answers_when_not_ready(client):
    client.stub._ready = False
    body = client.get('/health').json()
    assert body['status'] == 'loading'
    assert body['ready'] is False


# ---------------------------------------------------------------------- CORS
#
# Only needed for a split deploy (UI on a static host, API elsewhere). These
# tests build their own app because CORS is wired at construction time.

def _app_with_origins(origins, tmp_path):
    from app.config import Settings
    from app.main import create_app

    return create_app(Settings(data_dir=tmp_path / 'd', cors_origins=origins))


def test_cors_is_off_by_default(tmp_path):
    """A same-origin deploy must not advertise cross-origin access."""
    from fastapi.testclient import TestClient

    app = _app_with_origins([], tmp_path)
    with TestClient(app) as client:
        response = client.get('/health', headers={'Origin': 'https://evil.example'})
        assert 'access-control-allow-origin' not in response.headers


def test_cors_allows_a_configured_origin(tmp_path):
    from fastapi.testclient import TestClient

    app = _app_with_origins(['https://ui.example.app'], tmp_path)
    with TestClient(app) as client:
        response = client.get('/health', headers={'Origin': 'https://ui.example.app'})
        assert response.headers['access-control-allow-origin'] == 'https://ui.example.app'


def test_cors_rejects_an_unlisted_origin(tmp_path):
    from fastapi.testclient import TestClient

    app = _app_with_origins(['https://ui.example.app'], tmp_path)
    with TestClient(app) as client:
        response = client.get('/health', headers={'Origin': 'https://evil.example'})
        assert 'access-control-allow-origin' not in response.headers


def test_cors_preflight_permits_post(tmp_path):
    """The browser preflights the multipart upload; POST must be allowed."""
    from fastapi.testclient import TestClient

    app = _app_with_origins(['https://ui.example.app'], tmp_path)
    with TestClient(app) as client:
        response = client.options(
            '/detect',
            headers={
                'Origin': 'https://ui.example.app',
                'Access-Control-Request-Method': 'POST',
            },
        )
        assert response.status_code in (200, 204)
        assert 'POST' in response.headers['access-control-allow-methods']


def test_cors_origins_accepts_a_comma_separated_string():
    """Hosts hand env vars over as strings, not lists."""
    from app.config import Settings

    settings = Settings(cors_origins='https://a.app, https://b.app')
    assert settings.cors_origins == ['https://a.app', 'https://b.app']


def test_cors_wildcard(tmp_path):
    from fastapi.testclient import TestClient

    app = _app_with_origins(['*'], tmp_path)
    with TestClient(app) as client:
        response = client.get('/health', headers={'Origin': 'https://anything.example'})
        assert response.headers['access-control-allow-origin'] == '*'
