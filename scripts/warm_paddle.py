"""
Download the PaddleOCR models into the deployment bundle at build time.

PaddleOCR does not ship its weights inside the pip package -- it fetches them
the first time `PaddleOCR(...)` is constructed and caches them under
`~/.paddlex`. On a serverless host that is two separate problems:

  * `$HOME` is not writable, so the download fails outright; and
  * even where it succeeds, it would repeat on every cold start, adding a
    network round trip to a path that is already the slowest thing here.

So we fetch them once, during the build, into a directory that becomes part of
the function bundle. Only the pinned mobile pair is needed -- about 21 MB, versus
the ~230 MB the default server tier and document-analysis extras would drag in.

    PADDLE_MODEL_DIR=.paddlex python scripts/warm_paddle.py

At runtime `app/ocr.py` seeds a writable cache from this directory instead of
downloading anything. Exits non-zero if the models do not appear, so a silent
failure here cannot become a slow mystery in production.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep in step with the defaults in app/config.py.
DEFAULT_DET = 'PP-OCRv5_mobile_det'
DEFAULT_REC = 'PP-OCRv5_mobile_rec'


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def main() -> None:
    target = Path(os.environ.get('PADDLE_MODEL_DIR') or (PROJECT_ROOT / '.paddlex'))
    target = target if target.is_absolute() else (PROJECT_ROOT / target)
    target.mkdir(parents=True, exist_ok=True)

    det = os.environ.get('PADDLE_DETECTION_MODEL') or DEFAULT_DET
    rec = os.environ.get('PADDLE_RECOGNITION_MODEL') or DEFAULT_REC

    # paddlex reads this at import time, so it must be set before the import.
    os.environ['PADDLE_PDX_CACHE_HOME'] = str(target)

    print(f'warm_paddle: cache   {target}')
    print(f'warm_paddle: models  det={det} rec={rec}')

    import numpy as np
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        text_detection_model_name=det,
        text_recognition_model_name=rec,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # Construction downloads the weights, but a real prediction is what forces
    # every lazily-initialised sub-model to materialise. A blank frame is enough.
    ocr.predict(input=np.zeros((64, 192, 3), dtype=np.uint8))

    models = target / 'official_models'
    if not models.is_dir():
        print(f'warm_paddle: expected {models} to exist after warm-up',
              file=sys.stderr)
        raise SystemExit(1)

    present = sorted(item.name for item in models.iterdir() if item.is_dir())
    missing = [name for name in (det, rec) if name not in present]
    if missing:
        print(f'warm_paddle: missing {", ".join(missing)} (found: {present})',
              file=sys.stderr)
        raise SystemExit(1)

    print(f'warm_paddle: cached {len(present)} model(s): {", ".join(present)}')
    print(f'warm_paddle: {directory_size(target) / 1e6:.1f} MB in {target}')


if __name__ == '__main__':
    main()
