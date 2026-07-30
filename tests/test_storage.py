"""Artifact naming, retention and path safety."""

from __future__ import annotations

import os
import time

import pytest

from app.storage import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'detected', retention_hours=1)


def test_names_are_generated_and_unique(store):
    first = store.save_jpeg(b'\xff\xd8one')
    second = store.save_jpeg(b'\xff\xd8two')
    assert first != second
    assert store.path_for(first).read_bytes() == b'\xff\xd8one'
    assert store.path_for(second).read_bytes() == b'\xff\xd8two'


def test_client_filename_is_never_used(store):
    """Naming is entirely server-side, so a hostile name cannot influence it."""
    name = store.save_jpeg(b'\xff\xd8data')
    assert name.endswith('.jpg')
    assert len(name) == 36                     # 32 hex chars + ".jpg"
    assert all(c in '0123456789abcdef' for c in name[:-4])


def test_no_partial_files_are_left_behind(store):
    store.save_jpeg(b'\xff\xd8data')
    assert list(store.directory.glob('*.part')) == []


@pytest.mark.parametrize('name', [
    '../secret.jpg',
    '..\\secret.jpg',
    '/etc/passwd',
    'short.jpg',
    'nothex' + 'x' * 26 + '.jpg',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg',    # well-formed but absent
])
def test_path_for_rejects_unsafe_or_missing(store, name):
    assert store.path_for(name) is None


def test_prune_removes_only_old_files(store):
    fresh = store.save_jpeg(b'\xff\xd8fresh')
    stale = store.save_jpeg(b'\xff\xd8stale')

    old = time.time() - 2 * 3600
    os.utime(store.path_for(stale), (old, old))

    assert store.prune() == 1
    assert store.path_for(fresh) is not None
    assert not (store.directory / stale).exists()


def test_prune_disabled_when_retention_is_zero(tmp_path):
    store = ArtifactStore(tmp_path / 'd', retention_hours=0)
    name = store.save_jpeg(b'\xff\xd8data')
    old = time.time() - 10_000_000
    os.utime(store.path_for(name), (old, old))

    assert store.prune() == 0
    assert store.path_for(name) is not None


def test_prune_clears_abandoned_staging_files(store):
    orphan = store.directory / 'aborted.jpg.part'
    orphan.write_bytes(b'partial')
    old = time.time() - 2 * 3600
    os.utime(orphan, (old, old))

    assert store.prune() == 1
    assert not orphan.exists()


def test_directory_is_created_on_demand(tmp_path):
    target = tmp_path / 'nested' / 'deeper'
    assert not target.exists()
    ArtifactStore(target)
    assert target.is_dir()
