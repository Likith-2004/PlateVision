# Deployment

PlateVision ships as **one Vercel project** — one domain, no CORS. A `vercel.json`
[services][svc] block declares two halves:

```
                    ┌─ /api/*  ──► python service (FastAPI + YOLO + PaddleOCR)
Browser ──► Vercel ──┤
                    └─ /*      ──► static service (React bundle, on the CDN)
```

The React bundle is served from the CDN and loads instantly; only `/api/*`
invokes the Python function. This guide is the step-by-step; the README's
*Deployment* section is the summary.

> **Self-hosting needs none of this.** `API_PREFIX` and `INLINE_IMAGES` default
> off, so `uvicorn app.main:app` serves the UI and API together at the root. The
> steps below exist only because a serverless host imposes constraints a normal
> server does not.

---

## Ship the weights (optional)

`model/best.pt` (~52 MB) is gitignored — large binaries bloat git history
permanently — so a clean clone, which is exactly what a Vercel build starts
from, has no weights. [`scripts/fetch_model.py`](scripts/fetch_model.py) runs at
build time; if `MODEL_URL` is set it downloads the weights, and if it isn't the
step is **skipped so the build still succeeds**. A deploy with no weights comes
up fine — the frontend and API work — and `/health` simply reports
`model_loaded: false` until weights are provided. Detection endpoints need the
model, so add it before relying on `/detect`.

You have three ways to get the weights onto the deployment; pick one:

- **`MODEL_URL`** — host `best.pt` at a **direct-download URL** (one that returns
  the raw file, not an HTML page; a GitHub Release asset is simplest) and set
  `MODEL_URL` so the build fetches it. Steps below.
- **Commit the file** — drop the gitignore entry for `model/best.pt` and commit
  it. Simplest, but bloats the repo permanently.
- **Git LFS** — track `model/best.pt` with LFS to keep the file out of normal
  history.

The rest of this section covers the `MODEL_URL` route. Skip it entirely if you
are deploying without the model for now.

**1. Publish `model/best.pt` as a release asset.**

With the GitHub CLI, from the project root:

```bash
gh release create weights-v1 model/best.pt --title "Model weights" --notes "YOLO plate detector best.pt"
```

Or via the web UI: **Releases → Draft a new release**, create a tag (e.g.
`weights-v1`), drag `model/best.pt` into *Attach binaries*, and publish.

The direct URL is then:

```
https://github.com/<owner>/<repo>/releases/download/<tag>/best.pt
```

For this repo that is:

```
https://github.com/Likith-2004/PlateVision/releases/download/weights-v1/best.pt
```

The plain release URL 302-redirects to a per-request signed CDN link, so the
release URL itself does not expire — it is safe to hard-code as a build input.

**2. Compute the checksum** so the build can verify the download:

```bash
# Linux / macOS
sha256sum model/best.pt
# Windows PowerShell
Get-FileHash model\best.pt -Algorithm SHA256
```

`fetch_model.py` rejects anything under 1 MB (a served error page or a Git LFS
pointer instead of the file) and, when `MODEL_SHA256` is set, aborts on a
checksum mismatch — so a bad upload fails the build instead of shipping broken
weights.

---

## Environment variables

There are two kinds, and they are supplied differently.

**Build-time (optional)** — read by `scripts/fetch_model.py` while the bundle is
built. Set these **only if** you are using the `MODEL_URL` route from *Ship the
weights* above; leave them unset to deploy without the model. Set them either in
the dashboard or by prefixing them in the `api` `buildCommand` in `vercel.json`:

| Variable | Value | Why |
|---|---|---|
| `MODEL_URL` | the release-asset URL | Where `fetch_model.py` downloads the weights |
| `MODEL_SHA256` | the `sha256sum` of `best.pt` | Verifies the download; strongly recommended |

**Runtime** — read by the app ([app/config.py](app/config.py)) when a request is
served, so a build-command prefix never reaches them. Set these in **Vercel →
Project → Settings → Environment Variables**, scoped to **Production** (and
Preview if you deploy preview branches):

| Variable | Value | Why |
|---|---|---|
| `API_PREFIX` | `/api` | Routes must live under `/api` — see below |
| `INLINE_IMAGES` | `true` | The serverless filesystem is ephemeral — see below |
| `PADDLE_MODEL_DIR` | `.paddlex` | Where `warm_paddle.py` baked the OCR weights — see below |
| `PADDLE_CACHE_DIR` | `/tmp/paddlex` | Writable cache seeded from the baked copy — see below |

All four runtime settings default to off/unset, which is correct for a local
run; each is only turned on for the deployment. The full annotated list of
application settings is in [.env.example](.env.example).

> **Scope matters.** A push to `main` is a **Production** deploy, so a variable
> saved only for Preview (or not saved at all) is invisible to it — the build or
> the running app then behaves as if it were unset. If something reads as
> "missing" despite being entered, check the environment checkboxes first.

### Why `API_PREFIX=/api`

`vercel.json` rewrites `/api/*` to the Python service, and the service receives
the **original** path — so the FastAPI routes must actually live under `/api`.
`API_PREFIX=/api` moves them there. It must match the frontend's
`VITE_API_BASE`, which the Vercel build already sets to `/api` via
`frontend/.env.vercel` (loaded by `npm run build:vercel`'s `--mode vercel`). If
these disagree, the page loads perfectly while every request 404s.

### Why `INLINE_IMAGES=true`

By default `/detect` writes the annotated image to disk and returns a
`/media/<id>.jpg` URL to fetch separately. On a serverless host the filesystem
is ephemeral **and** per-instance, so the browser's follow-up GET can land on an
instance that never wrote the file — an image that works most of the time, which
is worse than one that never does. `INLINE_IMAGES=true` returns the image inline
as a data URI in the JSON response instead, at ~33% base64 overhead on a
response normally well under 1 MB.

### Why `PADDLE_MODEL_DIR` and `PADDLE_CACHE_DIR`

`scripts/warm_paddle.py` bakes the OCR weights into `.paddlex` at build time, but
baking them in is only half the job — the app has to be told to use them, or it
falls back to PaddleOCR's default of downloading into `~/.paddlex` on first use.
On a serverless host `$HOME` is read-only, so that download fails and OCR never
loads. `PADDLE_MODEL_DIR=.paddlex` points the app at the baked copy, and
`PADDLE_CACHE_DIR=/tmp/paddlex` gives it a writable location to seed from that
copy once per process (`/tmp` is the writable path on Vercel). Set neither and
the warm-up work is wasted; set only the model dir and there is nowhere writable
to seed into. See [app/ocr.py](app/ocr.py) `_prepare_cache`.

---

## Install: CPU-only torch

`vercel.json` installs the API service from **`requirements-deploy.txt`**, not
`requirements.txt`. This matters: on Linux, `pip install torch` from PyPI pulls
the CUDA build — ~2 GB of GPU libraries this project never uses — which is the
single most common reason these builds fail. `requirements-deploy.txt` pins the
CPU-only wheels (`torch==2.7.1+cpu`, ~200 MB) from the PyTorch CPU index, and
omits EasyOCR since PaddleOCR is the default engine.

This became feasible in June 2026, when Vercel raised the Python bundle ceiling
to **5 GB** ([large functions][lf], auto-enabled for new projects). The older
250 MB limit could not fit `torch` at all.

`scripts/warm_paddle.py` then bakes the pinned mobile OCR models (~22 MB) into
the bundle, because PaddleOCR otherwise fetches them on first use into a
read-only `$HOME` — which fails outright on serverless, or re-downloads on every
cold start where it doesn't.

---

## Deploy

With the release published and the four variables set, trigger a deploy (push to
the connected branch, or **Redeploy** in the dashboard). The build runs:

```
python scripts/fetch_model.py && python scripts/warm_paddle.py
```

`fetch_model.py` downloads and checksums the weights; `warm_paddle.py` caches
the OCR models. A successful build ends with both scripts reporting the sizes
they wrote.

### Verify

```bash
curl https://<your-deployment>/api/health
```

```json
{ "status": "ok", "ready": true, "model_loaded": true,
  "ocr_loaded": true, "version": "2.0.0", "detail": null }
```

`model_loaded: true` confirms the weights fetch worked; `ocr_loaded: true`
confirms the Paddle warm-up did. If either is false, `detail` says why. Then
send a real image:

```bash
curl -F "image=@car.jpg" https://<your-deployment>/api/detect
```

---

## Platform constraints, and where each is handled

| Constraint | Consequence | Handled by |
|---|---|---|
| Weights are gitignored | Nothing to load on a clean clone | `scripts/fetch_model.py` fetches from `MODEL_URL` if set, else skips (deploy runs without the model) |
| PaddleOCR fetches weights on first use | Read-only `$HOME`; re-download per cold start | `scripts/warm_paddle.py` bakes in ~22 MB; `PADDLE_MODEL_DIR` / `PADDLE_CACHE_DIR` wire it up at runtime |
| Filesystem is ephemeral and per-instance | `/media/<id>.jpg` 404s intermittently | `INLINE_IMAGES=true` → data URI |
| Request body capped at 4.5 MB | Phone photos rejected at the edge | `frontend/src/image.ts` resizes before upload |
| Torch defaults to the CUDA build | ~2 GB, exceeds sane limits | `requirements-deploy.txt` pins `+cpu` |

---

## Cold starts

Vercel scales to zero, so the first detection after an idle period pays the
bundle-load and model-init cost (roughly 30–60 s) before it answers. Options:

- **Accept it.** Fine for a demo or low-traffic use; every subsequent request on
  a warm instance is fast.
- **Keep one warm.** Ping `GET /api/health` on a schedule (e.g. a cron job every
  few minutes). `/health` does not touch the models, so it is cheap and does not
  itself trigger inference.
- **Use an always-warm host.** If cold starts matter more than a single free
  deployment, a container host (Fly.io, Render, Railway, a small VM) that keeps
  the process resident avoids the problem entirely. Self-hosting is unchanged —
  `uvicorn app.main:app` with the same weights at `model/best.pt`.

---

## Operational notes

- **Keep `--workers 1`.** Each worker loads its own ~950 MB copy of the models,
  so two on a 2 GB box will be OOM-killed.
- **No rate limiting or authentication ships.** Each `/detect` call costs seconds
  of CPU — add rate limiting before exposing it publicly.
- **Lower `OCR_MAX_PASSES=1`** to cap latency at roughly the detection cost, at
  the price of failing on marginal plates.

[svc]: https://vercel.com/docs/services
[lf]: https://vercel.com/docs/functions/limitations#large-functions
