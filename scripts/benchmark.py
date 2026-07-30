#!/usr/bin/env python
"""
Measure recognition accuracy and latency over a labelled image set.

This exists because tuning OCR without measurement is guesswork. Point it at a
directory of images plus a CSV of ground truth and it reports exact-match
accuracy, so a preprocessing or threshold change can be judged instead of
assumed.

    # 1. label some images
    #    labels.csv:  filename,plate
    #                 car01.jpg,MH12AB1234
    #                 car02.jpg,22BH6517AA
    #
    # 2. measure the current pipeline
    python scripts/benchmark.py --images data/eval --labels data/eval/labels.csv
    #
    # 3. change one thing, measure again, compare
    python scripts/benchmark.py --images data/eval --labels ... --max-passes 1

Images with no label are still processed and reported, but excluded from the
accuracy figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings                     # noqa: E402
from app.inference import PlateRecognizer, decode_image  # noqa: E402


def load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels: dict[str, str] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            name = (row.get('filename') or '').strip()
            plate = (row.get('plate') or '').strip().upper()
            if name and plate:
                labels[name] = plate
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, required=True,
                        help='Directory of images to evaluate.')
    parser.add_argument('--labels', type=Path,
                        help='CSV with filename,plate columns.')
    parser.add_argument('--max-passes', type=int,
                        help='Override OCR_MAX_PASSES for this run.')
    parser.add_argument('--time-budget', type=float,
                        help='Override OCR_TIME_BUDGET for this run.')
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f'{args.images} is not a directory')

    overrides = {}
    if args.max_passes is not None:
        overrides['ocr_max_passes'] = args.max_passes
    if args.time_budget is not None:
        overrides['ocr_time_budget'] = args.time_budget
    settings = Settings(**overrides)

    labels = load_labels(args.labels)
    files = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in settings.allowed_extensions and p.stat().st_size > 0
    )
    if not files:
        print(f'No usable images in {args.images}')
        return 1

    print(f'Loading models (max_passes={settings.ocr_max_passes}, '
          f'time_budget={settings.ocr_time_budget}s)...')
    recognizer = PlateRecognizer(settings)
    recognizer.load()

    correct = labelled = detected = verified = 0
    durations: list[float] = []
    misses: list[tuple[str, str, str]] = []

    print(f'\n{"image":<28} {"expected":<13} {"got":<13} {"":<9} {"ms":>7}')
    print('-' * 76)

    for path in files:
        started = time.monotonic()
        try:
            outcome = recognizer.detect(decode_image(path.read_bytes()), annotate=False)
        except Exception as exc:
            print(f'{path.name:<28} {"":<13} {"ERROR":<13} {exc}')
            continue
        elapsed = (time.monotonic() - started) * 1000
        durations.append(elapsed)

        best = max(
            (p.reading for p in outcome.plates if p.reading),
            key=lambda r: (r['valid'], r['confidence']),
            default=None,
        )
        got = best['number'] if best else ''
        if outcome.plates:
            detected += 1
        if best and best['valid']:
            verified += 1

        expected = labels.get(path.name, '')
        if expected:
            labelled += 1
            hit = got == expected
            correct += hit
            mark = 'ok' if hit else 'WRONG'
            if not hit:
                misses.append((path.name, expected, got or '(none)'))
        else:
            mark = '-'

        print(f'{path.name[:27]:<28} {expected or "-":<13} {got or "-":<13} '
              f'{mark:<9} {elapsed:>7.0f}')

    print('-' * 76)
    print(f'images          {len(durations)}')
    print(f'plate detected  {detected} ({detected / max(1, len(durations)):.0%})')
    print(f'format-valid    {verified} ({verified / max(1, len(durations)):.0%})')
    if labelled:
        print(f'exact match     {correct}/{labelled} ({correct / labelled:.1%})')
    else:
        print('exact match     n/a - no labels supplied, so accuracy is unknown')
    if durations:
        ordered = sorted(durations)
        print(f'latency         mean {sum(durations) / len(durations):.0f}ms  '
              f'median {ordered[len(ordered) // 2]:.0f}ms  max {ordered[-1]:.0f}ms')

    if misses:
        print('\nmismatches:')
        for name, expected, got in misses:
            print(f'  {name}: expected {expected}, got {got}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
