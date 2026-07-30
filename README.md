# PlateVision

Indian vehicle number plate detection and recognition. A fine-tuned YOLO model
locates plates; EasyOCR reads them; a format resolver turns the raw OCR text
into an actual plate number — or admits that it could not.

**Stack:** FastAPI · Pydantic v2 · React · TypeScript · Vite · Ultralytics YOLO · PaddleOCR / EasyOCR · OpenCV · pytest

---

## Why the `valid` flag matters

OCR on a plate crop returns whatever text it finds: the plate, dealer
watermarks, the word "IND", partial reads, and the usual `O`/`0`, `I`/`1`,
`S`/`5` confusions — with no indication of which is which.

Picking the highest-confidence string gets this wrong. On one of the sample
images, EasyOCR reads a dealer watermark `alam` at **97.6%** and the actual
plate `22 BH6517` at **43.4%**. Confidence alone returns `ALAM`.

So every reading is resolved against real Indian plate formats — the 1988
series (`MH12AB1234`) and the Bharat series (`22BH6517AA`) — including a check
that the state code actually exists. A reading that matches a format beats one
that does not, regardless of confidence.

Readings therefore come back in two kinds:

| `valid` | Meaning |
|---|---|
| `true`  | Matched a known plate format. Trustworthy. |
| `false` | OCR produced something unrecognisable. Returned **as-is**, flagged, and never presented as confirmed. |

The pipeline does not invent characters to force a match. Confusion repair is
capped at 30% of the string — beyond that it would be fabrication. `QQQQQQQQ`
can be forced into the valid-looking `QQ0Q0000` with five edits, so it is
rejected instead.

### Crop the plate generously

Detector boxes hug the glyphs, so the margin added before OCR has to scale with
the plate. A fixed 10 px margin clipped the outermost character and produced
**silently shortened readings** — `22BH6517A` read as `22BH6517`, and
`CG04H8801` as `CG04H880`, which matches no format and looks like a damaged
plate rather than a cropping bug.

Two independent OCR engines agreed on the truncated text at 98–99% confidence,
which is a good reminder that engine agreement says nothing about whether the
*input* was right. `CROP_PADDING_RATIO` defaults to 6% of the box's longer side.

## Choosing an OCR engine

Two backends ship behind one setting, `OCR_ENGINE`. Measured on the sample
plates — same YOLO crops, same resolver, single pass:

| | EasyOCR | Paddle *server* tier | **Paddle *mobile*** (default) |
|---|---|---|---|
| load | 12.2s | 208.1s | 19.6s |
| per plate | 1.4–3.1s | 13–15s | **0.29–0.41s** |
| clean plate | `22BH6517` 82% ✓ | `22BH6517A` 99% ✓ | **`22BH6517A` 99% ✓** |
| low-res plate | **✗** invalid, 23% | `HR26DO5551` 95% ✓ | **`HR26DO5551` 99% ✓** |
| installed size | ~109 MB | ~603 MB | ~603 MB |

Measured before the crop-padding fix above, so the absolute readings are now
better than shown; the *relative* standing is unchanged.

PaddleOCR mobile is both **4–8x faster and markedly more accurate** here, which
is why it is the default. Two things worth knowing:

- **The model tier matters enormously.** PaddleOCR 3.x picks server-grade models
  unless told otherwise — ~40x slower on CPU for no accuracy gain on plate
  crops. `PADDLE_DETECTION_MODEL` / `PADDLE_RECOGNITION_MODEL` pin the mobile
  tier; don't change them without measuring.
- **PaddleOCR reads a character EasyOCR drops.** On the clean sample it returns
  `22BH6517A` — including the trailing Bharat-series letter that EasyOCR misses
  entirely. The format is `##BH####XX`, so that letter is real.

Set `OCR_ENGINE=easyocr` to switch back; nothing else changes, because
`app/plates.py` never imports an engine. Both are pinned in `requirements.txt`,
and either can be dropped if you commit to the other.

> Caveat: this is **three images**. The margin is far too large to be noise, but
> "better on your real traffic" is unproven — see *Measuring accuracy* below.

## The OCR cascade

The cascade exists to squeeze readings out of a weak OCR pass, and it earns its
keep with EasyOCR. **With PaddleOCR it rarely fires at all** — pass 1 clears the
acceptance bar at ~99% on every sample — so treat the rest of this section as
the EasyOCR story, and as insurance for whatever image finally defeats Paddle.

Upscaling the crop is the single biggest accuracy lever, but it costs ~4x the
time (double the width is quadruple the pixels). No one variant wins on every
image, measured with EasyOCR on the samples:

| variant | clean plate | low-res plate | third sample |
|---|---|---|---|
| `clahe` (native) | 81.7% ✓ | 22.7% ✗ | 44.9% ✗ |
| `clahe+up2x` | 96.2% ✓ | 25.5% ✓ | 24.6% ✗ |
| `clahe+up3x` | 97.1% ✓ | 31.3% ✓ | 26.7% ✗ |
| `otsu+up2x` | 76.7% ✓ | 28.0% ✓ | 50.5% ✗ |

Since a correct *number* is what matters and the confidence figure is
cosmetic, the cascade runs **cheapest first** and stops as soon as a pass
yields a confident format-valid plate:

1. `clahe` — native resolution
2. `clahe+up2x`
3. `otsu+up2x`
4. `clahe+up3x`
5. `gray` — no CLAHE, which is what amplifies background watermarks

A clean plate costs exactly one native pass. Only crops that actually fail pay
for upscaling. If no pass clears the acceptance bar, every format-valid reading
is pooled and voted on, weighting agreement between variants above any single
confidence score. `OCR_TIME_BUDGET` caps the total so an unreadable plate
cannot stall a request.

Measured on one CPU (no GPU), whole request via `/detect`, with PaddleOCR:

| sample | result | passes | total |
|---|---|---|---|
| clean plate | `22BH6517A` valid | 1 | ~4.4s |
| low-res plate | `HR26DO5551` valid | 1 | ~3.9s |
| third sample | `CG04H8801` valid | 1 | ~9.2s |

Those were taken on a 2-core 1.6 GHz laptop sitting at ~85% CPU load from
unrelated processes, so treat them as an upper bound rather than the hardware's
capability — the same code measured 1.5–1.8s for a clean plate on the same
machine when it was idle. Timings here move by 4–5x with ambient load.

Roughly 1–2.7s of every figure above is YOLO detection, which runs before any
OCR. With PaddleOCR all three samples now resolve on the first, cheapest pass,
so the cascade adds nothing in the common case.

> **The time budget makes escalated output load-dependent.** When a crop does
> need later passes, a loaded machine may stop earlier than an idle one and
> return a different (still format-valid) reading. Set `OCR_TIME_BUDGET=0` for
> reproducible output at the cost of an unbounded worst case.

> **A skipped pass used to be invisible.** If a preprocessing variant raised,
> the cascade silently moved on, so a failure in pass 1 quietly demoted the
> answer to a later variant's reading with nothing in the logs. Variant
> failures are now logged with a traceback.

---

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # optional; every default is already correct

uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000> for the UI, or
<http://127.0.0.1:8000/docs> for interactive API documentation.

First start takes ~10–20s: the OCR engine fetches its recognition weights on
first use, and both models load. The server binds its port immediately and
`/health` reports `ready: false` until loading finishes, so the UI shows a
"loading models" state rather than failing to connect.

### Frontend

The UI is a React + TypeScript app in [frontend/](frontend/). Its production
build is committed to `app/web/`, so a fresh clone serves a working interface
with **no Node required** — just `uvicorn`.

To change the UI:

```bash
npm --prefix frontend install
npm --prefix frontend run dev      # localhost:5173, proxies the API to :8000
npm --prefix frontend run build    # rebuilds app/web/
```

`npm run build` type-checks first (`tsc --noEmit`), so a type error fails the
build rather than shipping. Edit `frontend/src` — `app/web/` is generated output
and hand edits there are lost on the next build.

`frontend/src/types.ts` mirrors `app/schemas.py`; if you change a response
model, update both. A test asserts the hashed asset filenames in `index.html`
actually resolve, since a stale reference produces a blank page with only a
console error.

### Model weights

`model/best.pt` (~52 MB) is **not** in git — large binaries bloat history
permanently. Obtain it via Git LFS or a release asset and place it at
`model/best.pt`, or point `MODEL_PATH` elsewhere.

Without it the server still starts, and `/health` reports the problem instead
of crash-looping.

---

## API

Full schema at `/docs`. Everything below is generated from the Pydantic models
in `app/schemas.py`, so it cannot drift from the implementation.

### `POST /detect`

Multipart upload, field name `image`.

```bash
curl -F "image=@car.jpg" http://127.0.0.1:8000/detect
```

```json
{
  "plates": [
    {
      "number": "22BH6517",
      "confidence": 81.68,
      "detection_confidence": 83.8,
      "valid": true,
      "format": "bh_series",
      "state": null,
      "corrections": 0,
      "variant": "clahe",
      "passes": 1,
      "votes": 1,
      "box": { "x1": 49, "y1": 39, "x2": 389, "y2": 160 }
    }
  ],
  "image_url": "/media/8d2f...c1.jpg",
  "source_width": 429,
  "source_height": 254,
  "elapsed_ms": 612.4
}
```

`box` coordinates are relative to the returned annotated image, which may have
been downscaled to `MAX_INFERENCE_EDGE`.

### `POST /detect/frame`

Same contract, tuned for live camera frames: one cheap OCR pass, and no
artifact written to disk. Send frames from the browser — the camera belongs
client-side.

### `GET /media/{name}`

The annotated image from a `/detect` call. Names are server-generated UUIDs.

### `GET /health`

```json
{ "status": "ok", "ready": true, "model_loaded": true,
  "ocr_loaded": true, "version": "2.0.0", "detail": null }
```

### Errors

Every non-2xx response has the same shape, with a stable machine-readable code:

```json
{ "error": "unsupported_media_type",
  "detail": "Unsupported file type \".md\".",
  "allowed": [".bmp", ".jpeg", ".jpg", ".png", ".webp"] }
```

| Status | `error` | Cause |
|---|---|---|
| 400 | `empty_file` / `undecodable_image` | Not a decodable image |
| 413 | `payload_too_large` | Exceeds `MAX_UPLOAD_MB` |
| 415 | `unsupported_media_type` | Extension not allowed |
| 422 | `validation_error` | Missing the `image` field |
| 503 | `not_ready` | Models still loading |

---

## Configuration

Environment variables, or a `.env` file. Parsed and validated by
`app/config.py` at startup, so a bad value fails immediately with a clear
message. See [.env.example](.env.example) for the annotated full list.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address |
| `MODEL_PATH` | `model/best.pt` | YOLO weights |
| `DATA_DIR` | `data` | Uploads and annotated output |
| `MAX_UPLOAD_MB` | `10` | Upload ceiling |
| `RETENTION_HOURS` | `24` | Prune artifacts older than this; `0` disables |
| `OCR_ENGINE` | `paddleocr` | `paddleocr` or `easyocr` |
| `PADDLE_DETECTION_MODEL` | `PP-OCRv5_mobile_det` | Paddle detector; mobile tier |
| `PADDLE_RECOGNITION_MODEL` | `PP-OCRv5_mobile_rec` | Paddle recogniser; mobile tier |
| `DETECTION_CONFIDENCE` | `0.5` | YOLO threshold — that a region *is* a plate |
| `MAX_INFERENCE_EDGE` | `1280` | Downscale longest edge before detection |
| `CROP_PADDING_RATIO` | `0.06` | Crop margin as a fraction of the box's longer side |
| `CROP_PADDING_MIN` | `8` | Pixel floor for that margin |
| `OCR_MIN_CONFIDENCE` | `0.1` | Floor for text that matches **no** format |
| `OCR_ACCEPT_CONFIDENCE` | `0.70` | Confidence that ends the cascade early |
| `OCR_MAX_PASSES` | `3` | Preprocessing passes; `1` = single native pass |
| `OCR_TIME_BUDGET` | `4.0` | Seconds ceiling on OCR per plate |
| `OCR_LIVE_MAX_PASSES` | `1` | Passes for `/detect/frame` |

`DETECTION_CONFIDENCE` and `OCR_MIN_CONFIDENCE` measure different things — how
sure YOLO is that a region is a plate, versus how sure OCR is of the text — and
are deliberately separate knobs.

---

## Project layout

```
app/
  config.py       Typed settings (pydantic-settings), validated at startup
  schemas.py      Pydantic response models — the API contract
  plates.py       Format resolution, confusion repair, OCR cascade
  ocr.py          PaddleOCR / EasyOCR backends behind one interface
  inference.py    YOLO + OCR backend, loaded once, lock-serialised
  storage.py      Artifact writing with UUID names and retention pruning
  routes.py       HTTP endpoints
  main.py         App factory, lifespan model loading, error handlers
  web/            Built React bundle (generated -- see frontend/)
frontend/
  src/
    App.tsx         Layout, health polling, request flow
    api.ts          Typed client; uniform ApiError
    types.ts        Mirrors app/schemas.py
    styles.css      Design system
    components/     UploadPanel, CameraPanel, Results, PlateCard, About
tests/
  test_plates.py       Format/cascade logic (no model needed)
  test_ocr.py          Backend selection + output normalisation
  test_cropping.py     Crop padding around detections
  test_api.py          HTTP contract against a stubbed recognizer
  test_storage.py      Naming, retention, path-traversal safety
  test_integration.py  Real weights end-to-end (marked `slow`)
  samples/             Fixture images
scripts/
  benchmark.py    Accuracy + latency over a labelled set
model/best.pt     YOLO weights (not in git)
```

`app/plates.py` deliberately has no FastAPI, settings or model imports — it
takes OCR tuples in and returns dicts out, so it is unit-testable in
milliseconds without loading 52 MB of weights.

Inference is blocking and CPU-bound, so every call is dispatched to a worker
thread. The event loop stays free: a request spending 4s in OCR does not stop
others being accepted, and `/health` keeps answering throughout. Neither YOLO
nor EasyOCR is thread-safe, so model access is serialised behind a lock.

---

## Tests

```bash
pip install -e ".[dev]"

pytest -m "not slow"    # 112 tests, ~9s, no model required
pytest -m slow          # 10 tests against the real weights and both engines
pytest                  # everything

npm --prefix frontend run typecheck   # frontend types
```

The fast suite stubs the recognizer, so it covers validation, error handling and
response shape without loading a model. The slow suite skips automatically when
weights are absent.

## Measuring accuracy

The numbers in this README come from **three** sample images, which is an
anecdote, not a benchmark. Before tuning anything, build a labelled set:

```csv
filename,plate
car01.jpg,MH12AB1234
car02.jpg,22BH6517AA
```

```bash
python scripts/benchmark.py --images data/eval --labels data/eval/labels.csv
python scripts/benchmark.py --images data/eval --labels data/eval/labels.csv --max-passes 1
```

It reports exact-match accuracy plus mean/median/max latency, so a
preprocessing or threshold change can be judged rather than assumed. Roughly
50–100 labelled images makes the figures meaningful.

---

## Deployment

Step-by-step instructions are in **[DEPLOYMENT.md](DEPLOYMENT.md)**. The short
version:

Everything ships as **one Vercel project**, one domain, no CORS:

```
                    ┌─ /api/*  ──► python service (FastAPI + YOLO + PaddleOCR)
Browser ──► Vercel ──┤
                    └─ /*      ──► static service (React bundle, on the CDN)
```

`vercel.json` declares both halves as [services][svc]. The page is served from
the CDN and loads instantly; only `/api/*` invokes the Python function.

This became possible in June 2026, when Vercel raised the Python bundle ceiling
to **5 GB** ([large functions][lf], auto-enabled for new projects). The older
250 MB limit genuinely could not fit `torch`, and earlier revisions of this file
said so — that advice is obsolete.

Four things the platform forces, all handled in code:

| Constraint | Consequence | Handled by |
|---|---|---|
| Weights are gitignored | Nothing to load | `scripts/fetch_model.py` at build time |
| PaddleOCR fetches weights on first use | Read-only `$HOME`; re-download per cold start | `scripts/warm_paddle.py` bakes in 22 MB |
| Filesystem is ephemeral and per-instance | `/media/<id>.jpg` can 404 intermittently | `INLINE_IMAGES=true` → data URI |
| Request body capped at 4.5 MB | Phone photos rejected at the edge | `frontend/src/image.ts` resizes first |

Install from **`requirements-deploy.txt`** on Linux, not `requirements.txt`: it
pins CPU-only torch (~200 MB instead of ~2 GB of unused CUDA), the most common
reason these builds fail.

The one trade-off is **cold starts**. Vercel scales to zero, so the first
detection after an idle period pays ~30–60 s of bundle load plus model init.
`DEPLOYMENT.md` covers keeping an instance warm, and lists always-warm hosts if
that matters more than a single free deployment.

Self-hosting is unchanged and needs none of the above: `API_PREFIX` and
`INLINE_IMAGES` default off, so `uvicorn app.main:app` serves the UI and the API
together at the root.

[svc]: https://vercel.com/docs/services
[lf]: https://vercel.com/docs/functions/limitations#large-functions

### Operational notes

Keep **`--workers 1`**: each worker loads its own ~950 MB copy of the models, so
two on a 2 GB box will be OOM-killed. There is no rate limiting — add some
before exposing `/detect` publicly, since each call costs seconds of CPU.

Lower `OCR_MAX_PASSES` to `1` to cap latency at roughly the detection cost, at
the price of failing on marginal plates.

---

## Known limitations

- **Accuracy is unquantified.** Format validity is verified; exact-match
  accuracy against ground truth is not, pending a labelled set.
- **Single-line Latin plates only.** Two-line plates are recombined, but
  stacked/vertical layouts and non-Latin scripts are untested.
- **No perspective correction.** Plates at a sharp angle read poorly;
  deskewing before OCR is an obvious next improvement.
- **No rate limiting or authentication.**
