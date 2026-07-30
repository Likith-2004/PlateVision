"""
Settings parsing, especially from environment variables.

These matter disproportionately: a host supplies configuration as env strings,
so a field that only accepts a Python list crashes the process at startup with
`SettingsError` and no request ever reaches the app. Every collection setting
documented in .env.example is exercised in its documented form here.
"""

from __future__ import annotations

import pytest

from app.config import Settings


def env(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    # A local .env would otherwise leak into these assertions.
    return Settings(_env_file=None)


# ------------------------------------------------------------- collections
#
# Regression: pydantic-settings JSON-decodes collection fields before any
# validator runs, so CORS_ORIGINS=https://a.app raised SettingsError. Every
# case below crashed the app at import time.

@pytest.mark.parametrize('raw,expected', [
    ('https://a.app', ['https://a.app']),
    ('https://a.app,https://b.app', ['https://a.app', 'https://b.app']),
    ('https://a.app, https://b.app', ['https://a.app', 'https://b.app']),
    ('*', ['*']),
    ('["https://a.app","https://b.app"]', ['https://a.app', 'https://b.app']),
    ('', []),
])
def test_cors_origins_from_env(monkeypatch, raw, expected):
    assert env(monkeypatch, CORS_ORIGINS=raw).cors_origins == expected


@pytest.mark.parametrize('raw,expected', [
    ('en', ['en']),
    ('en,hi', ['en', 'hi']),
    ('en hi', ['en', 'hi']),
    ('["en","hi"]', ['en', 'hi']),
])
def test_ocr_languages_from_env(monkeypatch, raw, expected):
    assert env(monkeypatch, OCR_LANGUAGES=raw).ocr_languages == expected


@pytest.mark.parametrize('raw,expected', [
    ('.png,.jpg', {'.png', '.jpg'}),
    ('png,jpg', {'.png', '.jpg'}),
    ('PNG,JPG', {'.png', '.jpg'}),
    ('[".png",".JPG"]', {'.png', '.jpg'}),
])
def test_allowed_extensions_from_env(monkeypatch, raw, expected):
    assert env(monkeypatch, ALLOWED_EXTENSIONS=raw).allowed_extensions == expected


def test_defaults_need_no_environment():
    settings = Settings(_env_file=None)
    assert settings.cors_origins == []           # same-origin
    assert settings.ocr_languages == ['en']
    assert '.jpg' in settings.allowed_extensions


# ------------------------------------------------------------------ scalars

def test_scalars_from_env(monkeypatch):
    settings = env(
        monkeypatch,
        PORT='9001',
        OCR_ENGINE='easyocr',
        OCR_MAX_PASSES='1',
        OCR_TIME_BUDGET='2.5',
        CROP_PADDING_RATIO='0.1',
        MAX_UPLOAD_MB='25',
    )
    assert settings.port == 9001
    assert settings.ocr_engine == 'easyocr'
    assert settings.ocr_max_passes == 1
    assert settings.ocr_time_budget == 2.5
    assert settings.crop_padding_ratio == 0.1
    assert settings.max_upload_bytes == 25 * 1024 * 1024


@pytest.mark.parametrize('field,value', [
    ('detection_confidence', '1.5'),
    ('ocr_min_confidence', '-0.1'),
    ('ocr_accept_confidence', '2'),
])
def test_out_of_range_confidence_is_rejected(monkeypatch, field, value):
    """Fail loudly at startup rather than behaving oddly at request time."""
    monkeypatch.setenv(field.upper(), value)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_unknown_engine_is_rejected(monkeypatch):
    monkeypatch.setenv('OCR_ENGINE', 'tesseract')
    with pytest.raises(Exception):
        Settings(_env_file=None)


# -------------------------------------------------------------------- paths

def test_derived_paths_follow_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path / 'store', _env_file=None)
    assert settings.upload_dir == tmp_path / 'store' / 'uploads'
    assert settings.detected_dir == tmp_path / 'store' / 'detected'


def test_ensure_dirs_creates_them(tmp_path):
    settings = Settings(data_dir=tmp_path / 'store', _env_file=None)
    settings.ensure_dirs()
    assert settings.upload_dir.is_dir()
    assert settings.detected_dir.is_dir()
