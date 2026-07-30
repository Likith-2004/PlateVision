"""
Fetch the detector weights when they are not already in the working tree.

`model/*.pt` is gitignored -- 52 MB of binary does not belong in git history --
which means a build that starts from a clean clone has no weights. That is
exactly what a hosted deploy does, so without this step the deployed function
imports fine, fails to load the model, and reports the failure through /health
forever.

Run it as the deploy build command:

    python scripts/fetch_model.py

It is a no-op when the file already exists, so local development and repeated
builds never touch the network. Configure it with:

    MODEL_URL      direct download URL for best.pt (required when absent)
    MODEL_SHA256   expected checksum; strongly recommended
    MODEL_PATH     destination (default: model/best.pt)

Exits non-zero on any failure so a bad download fails the build loudly rather
than shipping a broken deployment.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / 'model' / 'best.pt'

# The real weights are ~52 MB. A few KB means the URL served an error page or an
# LFS pointer file rather than the model, which would otherwise surface much
# later as an unintelligible failure inside ultralytics.
MIN_PLAUSIBLE_BYTES = 1024 * 1024


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def fail(message: str) -> None:
    print(f'fetch_model: {message}', file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    target = Path(os.environ.get('MODEL_PATH') or DEFAULT_PATH)
    expected = (os.environ.get('MODEL_SHA256') or '').strip().lower()

    if target.is_file():
        size = target.stat().st_size
        if expected:
            actual = digest(target)
            if actual != expected:
                fail(
                    f'{target} is already present but its checksum does not match.\n'
                    f'  expected {expected}\n  actual   {actual}\n'
                    'Delete the file to re-download it.'
                )
        print(f'fetch_model: {target} already present ({size:,} bytes); nothing to do.')
        return

    url = (os.environ.get('MODEL_URL') or '').strip()
    if not url:
        fail(
            f'{target} is missing and MODEL_URL is not set.\n'
            'Publish model/best.pt as a downloadable asset and set MODEL_URL to '
            'its direct URL. See DEPLOYMENT.md > "Ship the weights".'
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    # Download to a staging name and rename, so an interrupted build cannot
    # leave a truncated file that looks like valid weights on the next run.
    staging = target.with_suffix(target.suffix + '.part')

    print(f'fetch_model: downloading {url}')
    try:
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
            if response.status != 200:
                fail(f'{url} returned HTTP {response.status}.')
            with staging.open('wb') as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
    except urllib.error.URLError as exc:
        staging.unlink(missing_ok=True)
        fail(f'could not download {url}: {exc}')

    size = staging.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        staging.unlink(missing_ok=True)
        fail(
            f'downloaded only {size:,} bytes, which is too small to be the '
            'weights. The URL probably served an error page, an HTML redirect, '
            'or a Git LFS pointer instead of the file itself.'
        )

    if expected:
        actual = digest(staging)
        if actual != expected:
            staging.unlink(missing_ok=True)
            fail(
                'checksum mismatch on the downloaded file.\n'
                f'  expected {expected}\n  actual   {actual}'
            )
        print('fetch_model: checksum verified.')
    else:
        print(
            'fetch_model: warning -- MODEL_SHA256 is not set, so the download '
            'was not verified.',
            file=sys.stderr,
        )

    staging.replace(target)
    print(f'fetch_model: wrote {target} ({size:,} bytes).')


if __name__ == '__main__':
    main()
