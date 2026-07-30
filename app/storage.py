"""
Artifact storage for annotated output images.

Two things the previous implementation got wrong are fixed here:

* Filenames are always generated server-side from a UUID. Reusing the client's
  filename meant two users uploading "plate.jpg" overwrote each other, and a
  concurrent request could read a half-written file.
* Old artifacts are pruned. Nothing cleaned up before, so the output directory
  grew without bound for the lifetime of the deployment.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Written filenames are always "<uuid4-hex>.jpg", so anything else in the
# directory was not produced by us and is left alone by pruning.
_NAME_LENGTH = 32


class ArtifactStore:
    """Writes annotated images and prunes stale ones."""

    def __init__(self, directory: Path, retention_hours: int = 24) -> None:
        self._dir = Path(directory)
        self._retention_hours = retention_hours
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    def save_jpeg(self, payload: bytes) -> str:
        """
        Write `payload` and return its generated filename.

        The write goes to a temporary name first and is then renamed, so a
        reader can never observe a partially-written file.
        """
        name = f'{uuid.uuid4().hex}.jpg'
        final = self._dir / name
        staging = final.with_suffix('.jpg.part')
        staging.write_bytes(payload)
        staging.replace(final)
        return name

    def path_for(self, name: str) -> Path | None:
        """
        Resolve a stored artifact by name.

        Rejects anything that is not a bare generated filename, so a crafted
        name cannot escape the directory.
        """
        if not name.endswith('.jpg'):
            return None
        stem = name[:-4]
        if len(stem) != _NAME_LENGTH or not all(
            c in '0123456789abcdef' for c in stem
        ):
            return None
        candidate = self._dir / name
        if not candidate.is_file():
            return None
        return candidate

    def prune(self) -> int:
        """Delete artifacts older than the retention window. Returns the count."""
        if self._retention_hours <= 0:
            return 0

        cutoff = time.time() - self._retention_hours * 3600
        removed = 0
        for entry in self._dir.glob('*.jpg'):
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError as exc:                      # pragma: no cover
                logger.warning('could not prune %s: %s', entry, exc)

        # Abandoned staging files from an interrupted write.
        for entry in self._dir.glob('*.jpg.part'):
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError:                             # pragma: no cover
                pass

        if removed:
            logger.info('pruned %d stale artifact(s)', removed)
        return removed
